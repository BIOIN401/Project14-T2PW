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
  *"a seam is not complete while its row is open"* — **the freeze seam is recorded incomplete.**
- **Owner** **C-030a** *(amended 2026-08-17; this line read **C-052**)*
- **Shortcut accepted** Deliberately. No vacuous assertion was written in its place.

> **Owner amended 2026-08-17 — F-008 and F-041 disagreed and both were standing.** F-008 recorded
> A0-C7's owner as **C-052**; **F-041** recorded the `LEDGER` owner column as **C-030** and A0-C7 as
> orphaned. The product owner ruled **F-041 authoritative**: **C-052 owns A0-C8 only and must not absorb
> A0-C7**, which is re-assigned to a new narrow follow-up card **`C-030a`**. This line is amended in place
> so the control plane no longer carries two answers to one question.
>
> **F-008's technical content is unaffected and remains load-bearing.** Its proof — that A0-C7 is
> **undischargeable inside `freeze_canonical_payload`**, because the canonical payload there is a
> `deepcopy` nothing in the returned dict aliases, so the share→copy mutant is indistinguishable from the
> tip on all nine seam observables across all 39 legs — is precisely why `C-030a` **cannot** inherit
> C-030's `MASTER_PLAN` §9 boundary. The real sharing sites sit in `run_post_pipeline_sbml_artifacts`
> (§9 row 1, shared with C-050 and C-052), so `C-030a` must declare exact functions/tests and serialize
> against C-052. Only the sentence *"The four real sharing sites are C-052's symbol"* is superseded: they
> are in a **jointly-owned** region, and being in it does not make the requirement C-052's.

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

---

## F-039 — `ir._dedupe_named_rows` still drops rows on a name-only collision, post-freeze

**Registered 2026-08-16** from the C-050h census, **verified independently by the orchestrator** against
live source. **This is not a re-opening of B-1; it is a distinct mutation class that B-1's accepted gate
did not measure.**

`_dedupe_named_rows` (`src/t2pw/pwml/ir.py:391-424`) groups rows on `_norm(_canonical(name))` and, on a
collision, **silently keeps the first and drops the rest**, emitting only a `duplicate_named_record`
*warning* (`:409-415`). It is called at **`:957`** and — on `entities.compounds` — at **`:1049`**.

**Why the accepted gate did not catch it.** The stack's landing measured *post-freeze compound identity
mutations*: **13 at the integration base, 0 at the merged tip**. That measurement is sound and is not
disturbed. But identity mutation (a row's `pathwhiz_id` / identifiers changing) and **row dropping** are
different classes. A first-wins dedupe removes a row without mutating any surviving row's identity, so it
passes an identity-mutation probe unseen.

**Status: OPEN, exposure unquantified.** Two things are *measured*: the function is live, and it drops on
name coincidence alone with no identity evidence — which is precisely what **D-035 clause 1** forbids. Two
things are **NOT yet measured and must not be asserted**:

1. whether `build_pwml_ir` runs **after** `freeze_canonical_payload` at each of the three export entry
   points (D-033), which is what decides whether this is an actual **merge rule 8** violation or a
   pre-freeze-adjacent dedupe;
2. how many committed legs currently reach it with a live collision — the D-034 leg does **not**, because
   it aborts earlier in pre-freeze.

**Do not state that merge rule 8 is violated until item 1 is measured.** Equally, do not treat the
question as closed by the B-1 discharge: B-1 measured a different thing.

**Owner: unassigned.** Natural candidate is **C-050h**, whose D-035 mandate already covers "never merge
rows because names coincide" — but the census also shows the two functions disagree about what a group
*is*, so folding it in without measuring item 1 would widen an unstudied boundary (**F-015**).

---

## F-040 — three disagreeing name normalizers, and a third name-only consolidator upstream

**Registered 2026-08-16** from the C-050h census.

A **third** name-only consolidator exists upstream of the pre-freeze stage:
`src/t2pw/pipeline/process_normalizer.py:706-721` merges compound rows on coincident `_normalize(name)`
with **zero identity evidence**, pre-freeze, reached from `:933`, `:1551` and `:1557`.

Three normalization keys are live and **mutually disagreeing**:

| Key | Site | Behaviour on the non-`[a-z0-9:+ ]` class |
|---|---|---|
| `_norm` | `pwml/prefreeze_resolution.py:94` | **substitutes a space** |
| `_normalize` | `pipeline/process_normalizer.py:333-335` | **deletes** |
| `_collapsed` | `pwml/prefreeze_resolution.py` | third variant |

Consequence: "the duplicate group" is not a well-defined set — it depends on which function asks. Any
D-035 implementation must **name the normalizer that defines a group** and say why, and must state whether
D-035 binds `process_normalizer._dedupe_named_rows` and `ir._dedupe_named_rows` (see **F-039**).

---

## F-041 — A0-C7 is orphaned: its owner card shipped without discharging it

**Registered 2026-08-16** by the orchestrator during C-052 charter validation.

`LEDGER.md`'s carried-requirement table records **A0-C7's owner as C-030** (C-052 is listed only under
"related"). **C-030 is `ACCEPTED`, merged `f3e9fb1`**, and its ledger cell records only **A0-C1** as
discharged. So A0-C7 — *"capture the pre-seam caller-owned payload reference and prove the intended
object-sharing relationship after `freeze_canonical_payload`; `final_mapped is result["payload"]` is
tautological and a share→copy mutant survives it"* — has **no live owner**.

**Correction to the sprint record:** the handoff note that "C-052 carries A0-C7 and A0-C8" is **half
wrong**. **A0-C8 is C-052's. A0-C7 is not**, and C-052 must not silently absorb it — that would let a
requirement change owner without a decision. A0-C7 needs an explicit re-assignment.

The C-052 study did locate a genuine discriminator for it — a **share→copy mutant** that the tautological
assertion survives — so re-assignment is actionable whenever an owner is named.

> **RESOLVED 2026-08-17.** The product owner ruled **F-041 authoritative** and re-assigned A0-C7 to a new
> narrow follow-up card **`C-030a`** (next unused suffix in the C-030 family; `C-052` is **not** broadened
> to absorb it). **F-008's conflicting `Owner: C-052` line is amended** so the control plane carries one
> answer. `C-030a` **cannot** inherit C-030's `MASTER_PLAN` §9 boundary (`freeze_canonical_payload`),
> because F-008 proved A0-C7 undischargeable inside it; the real sharing sites are in
> `run_post_pipeline_sbml_artifacts`, §9 row 1, **shared with C-050 and C-052**, so `C-030a` serializes
> against C-052 and declares exact functions/tests. See the `C-030a` row in `LEDGER.md`.
>
> **Line-number correction:** this finding recorded the discriminator at `streamlit_app.py:3748`.
> **Re-measured at `fd5afd8` it is `streamlit_app.py:3746`** — `"final_mapped": canonical_export_payload
> or final_export_payload`. The line had drifted by two. **The charter must use the measured number**,
> and any card quoting `:3748` is quoting a stale record.

---

## F-042 — `PATHSPEC` omits `scripts/`, so no exported base tree can run Chunk B or SMOKE

**Registered 2026-08-16** from the H-010 study, **verified independently by the orchestrator**, and
**deliberately excluded from H-010's boundary** so that card does not grow further.

`PATHSPEC` (`docs/pwml_recovery_sprint/evidence/c045b_base_tree.py:35-38`) is
`["src", "tests", "reference", "pytest.ini", "pyproject.toml", "data/pathwhiz_id_db.json",
"docs/pwml_recovery_sprint/evidence"]` — **`scripts/` is absent**.

**It affects the mandated materializer too:** `c051a_base_tree_batch.py:21` does
`from c045b_base_tree import PATHSPEC`, so every base tree built by the *required* tool is missing
`scripts/`.

**Blast radius exceeds imports.** Beyond `tests/test_bench_controls.py:344`'s
`from scripts import batch_run`, these reference `scripts/batch_run.py` **as a path**:
`tests/test_batch_preflight.py:57`/`:481`/`:590`/`:612`, `tests/test_batch_run.py:42`/`:1425`, and
`tests/helpers_eligibility.py:146-147` inserts `ROOT/"scripts"` on `sys.path` itself.

**Interaction to respect:** H-010's `require_scripts` guard defaults **off / selection-derived** precisely
so that it does not newly refuse Chunk D in an exported base tree while F-042 is unfixed. Fixing F-042 and
defaulting that guard on are the same decision and belong to one card.

---

## F-043 — a duplicate group that clears D-035's bar and is biologically wrong

**Registered 2026-08-16** from the C-050h census. **This is the strongest argument against
identifier-equality as a consolidation trigger.**

Three rows — `PG`, `PG phosphate`, `(PGP)` — all carry `pathbank_compound_id` **193**, satisfy every
D-035 clause 3 sub-test, and would consolidate. **PathBank 193 is UDP-glucose**, which is none of them.

Separately, in the D-034 leg, row 23 `lipid IV A` carries **row 6 `Kdo-lipid IV_A`'s exact identifier
triple** (pathbank 40738 / kegg C06025 / chebi CHEBI:60365). So the pair the old exporter silently merged
was **not** one molecule spelled two ways — it merged `lipid IV_A` with a row bearing a *different*
molecule's identity in a *different* biological state. **D-034's account of that leg understated the
error; the silent merge was worse than recorded, which strengthens rather than weakens D-034.**

**Consequences for any D-035 implementation:**

1. Consolidation must be **triggered by name collision** and only then *proved* by identity. Triggering on
   identifier equality is measurably unsafe.
2. Clause 3 must be evaluated on **payload-carried, pre-resolution** identifiers, with **3b (no conflicting
   identifiers) ordered before 3c (matching identifier)** — otherwise the `C_canned` stub stamps
   `pathbank_compound_id=78` onto **both** `PEtN-lipid A` and `modified Lipid A`
   (`db_resolver.py:458-472`) and **clause 7's must-keep-refusing reference case consolidates**.
3. Wrong-identity rows are a **separate mapping defect**, not a duplicate-row concern.

---

## F-039 — MEASURED AND RESOLVED · merge rule 8 **is** violated at EP3, and live exposure is **zero**

**Amended 2026-08-16** by the F-039 measurement. This supersedes F-039's "exposure unquantified" status.
The finding's own precondition — *do not assert a merge rule 8 violation until the call ordering is
measured* — **has now been discharged by measurement**, and the answer is that it **is** violated.

### Q1 — call ordering, per D-033 entry point

`freeze_canonical_payload` is defined at `streamlit_app.py:2509` and **called exactly once in production**,
at `streamlit_app.py:3627`.

| EP | Path | Verdict |
|---|---|---|
| **EP1** `run_post_pipeline_sbml_artifacts` `:2617` | `normalize_process_payload:2884` → `run_prefreeze_resolution:3587` → `freeze_canonical_payload:3627` → SBML `:3649` → return `:3680` | **Never reaches `build_pwml_ir` at all.** Freezes and serializes; builds no IR. **No rule-8 exposure.** |
| **EP2** CLI `writer.run_pwml_pipeline_export:2642` | `json.loads:2647` → `normalize_process_payload:2650` → `run_prefreeze_resolution:2692` → `build_pwml_ir:2734` | **`freeze_canonical_payload` is never called.** No in-process freeze, so before/after is undefined *in process*. Its documented input is `--in final.mapped.json` (`:2816`) — the serialization of EP1's frozen payload. **Post-freeze in the artifact sense, indeterminate in the process sense.** See open question 4 below. |
| **EP3** `run_pwml_export:3828` | `freeze_canonical_payload:3627` → `CANONICAL_PAYLOAD_KEY:3694` → UI `:5955` → `initialize_refinement_review_state:6033`/`:6059` → `deepcopy` → `refinement_working_json:1183` → button `:2228` → `_generate_pwml_from_refinement_working_json:1899` → `run_pwml_export:1939` → `deepcopy(final_payload):3869` → `run_prefreeze_resolution:4091` → `build_pwml_ir:4135` → `_dedupe_named_rows` (`ir.py:957`, `:1049`) | **DECISIVE — unambiguously post-freeze.** Operates on a deepcopy-of-a-deepcopy of the frozen payload, taken **after the hash** and after `tmp/final.canonical.json` was written. **MERGE RULE 8 IS VIOLATED HERE.** |

**EP3 is the path every committed batch leg takes** — `driver.py:129-130` binds `pwml_generate_btn` and
`refinement_generate_pwml`, and `driver.py:2085-2112` clicks the second after the first.

### Q2 — live exposure at the tip: **ZERO of 32 legs**

Replayed `ir._norm(ir._canonical(...))` (imported, not reimplemented) with first-wins across all 32
committed `final_mapped.json`, all five entity buckets (`ir.py:1003-1044`) and four component buckets
(`ir.py:923-933`). **Exactly one leg** ever collides — the D-034 leg, dropping `#23 'lipid IV A'` in favour
of `#5 'lipid IV_A'` — and **that leg now aborts pre-freeze at the tip**, pinned by
`tests/test_prefreeze_compound_resolution.py:1246`. The committed `pwml_ir.json` showing 44 rows in / 43 out
is a **base-era artifact**. All other 31 legs: zero collisions in every bucket.

### The concrete proof that this is invented biology, not a harmless duplicate

From committed artifacts alone, on that one leg: the frozen payload references `"lipid IV A"` in four
places including `/processes/reactions/9/inputs/0`, and frozen reaction 9 is
`"lipid IV A -> lipid A precursor"` with substrate **PathBank 40738 / ChEBI 60365 / KEGG C06025**. In
`pwml_ir.json`, reaction 9's `left` is `entity_key "cmp_6"` = `'lipid IV_A'`, **PathBank 40982 / ChEBI
58603**, and the string `"lipid IV A"` appears nowhere.

**The exporter re-bound a reaction to a different database compound after the hash, emitting one warning
and no error.**

References do **not** dangle — they **silently repoint**. `entity_by_name` (`ir.py:1105-1117`) is keyed on
the same `_norm` the dedupe grouped on, so `resolve_entity:1371` *succeeds* against the survivor and
`unresolved_entity_reference` never fires. Same for locations (`:1514`→`:1525`). Everything of the dropped
row is lost: `ir.py:408-417` warns then `continue`s — the dict is never copied, never entered in `by_norm`,
and its `synonyms` / `mapped_ids` / `raw_name` are not merged into the survivor.

### Status

**Rule-8 claim: CONFIRMED. Live corpus exposure: ZERO — this is hardening against a latent defect, not an
active data loss.** Routed as **C-050i**. Any card fixing it must be labelled **G9 new capability with an
explicitly labelled new acceptance test**: there is no base SHA at which a behavioural probe over the tip's
corpus fails, because the only leg that reaches the code now aborts earlier.

**Unmeasured residual — EXPLICITLY IN C-050i's SCOPE, assigned 2026-08-16.** Recorded because REV-050h observed this paragraph sits *after* the "Routed as C-050i" sentence and so inherited an owner **by proximity alone**. It is now assigned by name, so it cannot fall between C-050h and C-050i: C-050h creates the collision case it does not refuse (an *unreferenced* created duplicate, which exports identically at base and tip — measured), and C-050i owns the harm. EP3's *second* `run_prefreeze_resolution` can create a `_norm` collision absent
from the committed file. D-034 clause 5's structural blindness is caught today only when the collided row
is a **participant**; a non-participant compound in that shape still reaches the dedupe and drops silently.
Measuring it requires a live pre-freeze run — the heavy-job slot.

---

## F-040 — CORRECTED · two of the four "disagreeing" normalizers are byte-identical duplicates

**Amended 2026-08-16.** **F-040 as originally registered was half wrong and is corrected here rather than
left to mislead.**

`prefreeze_resolution._norm` (`:94-96`) and `._canonical` (`:90-91`) are **byte-identical duplicates** of
`ir._norm` (`ir.py:120-122`) and `ir._canonical` (`ir.py:116-117`) — verified by `inspect.getsource`
equality, and **deliberate**, per the comment at `prefreeze_resolution.py:83-87`. `ir` imports neither; it
defines its own copies. **They do not disagree.** The original finding's implication that `_norm` and the
pre-freeze normalizer are competing keys is **false**.

The genuine disagreement is between:

| Key | Site | Non-`[a-z0-9:+ ]` class |
|---|---|---|
| `_norm` (`ir` ≡ `prefreeze_resolution`) | `ir.py:120-122` | **substitutes a space** |
| `process_normalizer._normalize` | `:333-335` | **deletes** |
| `process_normalizer._norm_text` | `:338-341` | keeps `-`, drops `:`/`+` — **a fourth key F-040 originally omitted** |
| `_collapsed` | `prefreeze_resolution.py:743-747` | third variant, scoped to one comparison |

**`_norm` and `_normalize` are incomparable — neither is coarser:**

* `'lipid IV_A'` vs `'lipid IV A'` → `_norm` `'lipid iv a'` / `'lipid iv a'` **same group**; `_normalize`
  `'lipid iva'` / `'lipid iv a'` **different**. This is the real F-039 pair, and **this disagreement is the
  sole reason `ir._dedupe_named_rows` ever fires on committed data.**
* Reverse: `'UDP-GlcNAc'` vs `'UDPGlcNAc'` → `_normalize` same group, `_norm` different.

**Ruling: `_norm` is the only defensible definition of an exporter-relevant group**, because
`entity_by_name` (`ir.py:1105-1117`) and `resolve_entity` (`ir.py:1371`) key on it. Any card grouping rows
for exporter purposes uses `_norm` and says so.

---

## F-044 — `process_normalizer._dedupe_named_rows` merges 59 rows pre-freeze and records nothing

**Registered 2026-08-16** from the F-039 measurement. **Distinct from F-039**: opposite side of the freeze,
incomparable key, opposite failure mode.

`process_normalizer._dedupe_named_rows` (`:706-721`) is **pre-freeze** — reached from `_ensure_compound:922`
→ `:933` and `normalize_composites:1446` → `:1551-1559`; `normalize_composites` is called by
`normalize_process_payload:5426`, which Streamlit calls at `streamlit_app.py:2884` (743 lines *above* the
freeze, same function) and the CLI at `writer.py:2650`.

It merges on **`_normalize(name)` alone** (`:714`) via `_merge_dicts_keep_existing` (`:715-719`), with
`rows[:] = list(by_norm.values())` at `:721`. **No class check, no identifier comparison** — exactly what
**D-035 clause 1** forbids. It takes **no report argument and records nothing anywhere**: strictly more
silent than `ir._dedupe_named_rows`, which at least warns.

**Measured effect:** **8 of 60** committed `merged_payload.json` (`driver.py:1170-1172`, `:1204-1205`),
**59 rows merged away** — 5/3/1/1/21/22/2/4 across PMC12312563 ×2, PMC13231680 ×3, PMC12444477 ×2,
PMC13278307. **Lossiness today: zero** — for all 59 pairs, the set of fields where both rows hold a
non-blank *differing* value (excluding `evidence`, longest-wins at `:682-687`) is **empty**; they are
pre-mapping exact-name duplicates carrying no accessions. 0 of 32 `final_mapped.json` retain any
`_normalize` collision.

**It becomes lossy the moment a pair carries a conflict**, because `:700-701` only fills blanks. Being
pre-freeze it does not violate merge rule 8, and being currently lossless it is **not urgent** — but it is a
name-only consolidator operating with zero identity evidence and zero record, which is the exact practice
D-035 exists to forbid. **Owner: unassigned. Deliberately NOT folded into C-050h or C-050i** (F-015: that
would widen an unstudied boundary across three files and two sides of the freeze).

---

## F-045 — a measurement script resolved its output path AFTER `chdir`, writing evidence into the checkout it was auditing

**Registered 2026-08-16**, self-disclosed by H-010 during correction round 1 and **verified by the
orchestrator**. Fixed inside H-010; recorded because the *pattern* is not unique to it.

`pinned_pytest.py` resolved `--pin-verdict` **after** its `chdir`, so a **relative** verdict path combined
with `--expect-tree` wrote the artifact into the **other checkout**. H-010's own round-0 run 25 did exactly
this, creating an `evidence/g11/pin/` directory **inside the primary checkout** while measuring a worktree.

**Impact was contained:** the stray artifact was untracked, no tracked file in the primary was modified, and
the card moved it (same bytes, same process) into its worktree and removed the strays. Orchestrator-verified
afterwards: primary working tree is exactly the 7 protected scratch modifications + `.claude/settings.json`
+ the 3 protected untracked `topics_*.txt`; no `evidence/g11/pin` directory; no `*.pin.json` under the
primary's `g11/`; G11 **1870 / 0 non-compliant**, unchanged from takeover.

**Why it is recorded rather than closed silently.** A measurement harness that writes its own evidence into
whichever checkout it happens to be auditing is a defect in the same class the harness exists to prevent —
it corrupts the thing being measured, and in a sprint where **~70 worktrees are nested inside the primary
checkout** the blast radius is every one of them. It also produces evidence whose *location* misattributes
which tree a measurement belongs to.

**Residual, unowned:** any other evidence script that combines a `chdir` with a relative output path has the
same hazard. Not surveyed. A cheap sweep would look for `chdir` in `docs/pwml_recovery_sprint/evidence/*.py`
and check whether every output path is resolved before it.

**Rule to carry forward:** resolve every output path **before** any `chdir`, and prefer asserting the
artifact landed at the invocation cwd over trusting the path string.

## F-046 — the component-bucket residual left by C-050i's narrowed guard

**Registered 2026-08-17** by the orchestrator when C-050i hit R1's stop condition and correctly stopped
rather than weakening the guard or narrowing scope on its own authority.

**R1 as originally issued bound the post-freeze duplicate guard to *every* `_dedupe_named_rows` caller.
That was wrong for the component call site, and C-050i measured why.**

`prefreeze_resolution._canonicalize_species_rows` (`:1180-1197`) **deliberately converges a `_norm` group
onto its leader's name precisely because the exporter dedupe collapses it.** Its own docstring:

> *"`build_pwml_ir` canonicalized the output of `_dedupe_named_rows`, which keeps the **first** row of each
> `_norm(name)` group and drops the rest … a row that stops being a duplicate becomes a **second species**
> in the IR that the exporter never emitted. **That is inventing biology**, so the group leader is the row
> that gets canonicalized, and the rest of its group follows it **only when the leader's rename moved the
> group's `_norm`**, which is exactly the condition under which they would otherwise stop deduplicating."*

Two rows converged this way carry the **same `taxonomy_id`**: the collapse is the *intended* outcome of an
accepted pre-freeze identity resolution, not a spelling coincidence. **Refusing it would break C-045/D-016's
accepted acceptance criteria** — `tests/test_prefreeze_species_resolution.py:191` and `:312`, the second of
which exists precisely because *"a refusal on a path every species payload crosses is how a card turns a
working export into a dead one"* — **and would itself invent biology**, in the exact direction that module
was written to prevent.

**This is the D-035 distinction, in the two buckets.** F-039's compound pair (`lipid IV A` / `lipid IV_A`)
is **coincident spelling with conflicting identifiers** — PathBank 40738 vs 40982 — and the drop silently
repoints a reaction to a different molecule. The species pair is **proven identity**, deliberately produced.
D-035 permits a merge only on proven identity; the component case has it and the entity case does not.

**Ruling (see `LEDGER.md`): the guard binds the entity call site (`ir.py:1049`) only. The component call
site (`ir.py:957`) keeps its pre-existing warning.**

### The residual, named so it is not inherited by proximity

**A `_norm` collision in a component bucket that was NOT created by the pre-freeze converger still drops
first-wins with only a warning**, and `component_by_name` (`ir.py:995-1000`) is keyed on the same `_norm`,
so the same silent-repoint mechanism applies. Two genuinely different organisms whose names `_norm`-collide
would still lose one silently.

**Live exposure: zero.** C-050i re-measured all 32 committed legs across all nine buckets at its own tree and
found **exactly one** `_norm` collision — F-039's compound pair — independently confirming the F-039 census.
No component-bucket collision exists in the committed corpus other than the deliberate species convergence.

**Proposed discriminator, recorded so the design work is not lost.** `_canonicalize_species_rows` stamps a
**durable** marker (`ir.SPECIES_CANONICALIZATION_FIELD`) on *"every row that participated"*. A collision
whose rows all carry that marker is a deliberate convergence and may collapse; a collision whose rows do not
is a coincidence and should refuse. **This must NOT be built from identifier equality** — F-043 stands: `PG`,
`PG phosphate` and `(PGP)` all carry PathBank 193, which is UDP-glucose and wrong for all three, so equal
identifiers are not proof of identity.

**Owner: a new card `C-050j`. NOT C-050i** — assigned by name, not by proximity, because that is exactly the
failure REV-050h caught on F-039. **Not urgent**: exposure is measured at zero and the three non-species
component buckets have no converger at all.

## F-047 — reading the compound-resolution goldens outside pytest perturbs every IR-building leg

**Registered 2026-08-17** from C-050i's correction round, where it produced a **false 32-of-32 delta** and
was correctly diagnosed rather than worked around.

**Symptom.** Capturing `tests/test_compound_resolution_extraction.py::_leg_digest` by importing the test
module directly — the obvious way to measure a golden delta — reported **all 32 legs moved**. The granted
delta was one leg.

**Diagnosis, and why it is trustworthy.** The card measured the *same sweep at the base tree*, which also
reported all 32 moved. A base tree cannot have moved against itself, so the perturbation is in the
**instrument, not the code**. **pytest is the authority**; run under pytest, the sweep reports exactly the
one-leg delta.

> **Mechanism corrected 2026-08-17 by REV-050i — the phenomenon is real, the stated cause was not.**
> This finding first read: *"the digest folds the IR report through `json.dumps(..., default=_nonjson)`,
> i.e. `repr`, so any leg that builds an IR is sensitive to import-context differences."* The reviewer
> **instrumented `_nonjson` and it fires zero times** on the leg it measured, while independently
> reproducing the divergence (out-of-pytest digest `fc587e03…` ≠ pytest/`GOLDEN` `64038a74…` for an
> IR-building leg, with the moved leg agreeing exactly). **So the divergence is confirmed and its cause is
> NOT `_nonjson`; the true mechanism is unidentified.** The operational rule below is unaffected and the
> golden delta is proven two independent ways. Recorded rather than quietly reworded, because an
> authoritative-sounding wrong mechanism is worse than an admitted unknown — the next card would try to
> reason from it.

**Why the granted delta was still safe to accept.** The moved leg is *immune* to the hazard: after the move
**none of its five configs builds an IR**, so its digest is only config names plus stop/refusal codes — and
both instruments agreed on it exactly.

**Rule.** **Measure a `_leg_digest` delta under pytest, never by importing the test module.** If you must use
a direct-import harness, **measure the base tree with the same harness first** — agreement there is the only
thing that makes a delta claim meaningful. Recorded in-file above `GOLDEN` as well, so the next card meets it
where it works.

**Same family as F-045** (a measurement script resolved its output path after `chdir` and wrote evidence into
the checkout it was auditing): **the instrument is part of the measurement, and it is unsurveyed elsewhere.**
Unowned.

## F-048 — the alias residual binds first-wins by payload row order, and emits no warning at all

**Registered 2026-08-17** by REV-050i, which measured it rather than accepting the card's description.
**Pre-existing: base and tip behave identically**, so it was outside C-050i's boundary to fix — but the
card's own docstring described it **wrongly**, and a false record is the failure mode this sprint keeps
paying for.

**The card claimed** (`tests/test_pwml_ir_duplicate_row_refusal.py:462-463`) that an alias-only overlap
produces *"an ambiguous `entity_by_name` entry, resolved last-writer-wins … pinned here so the residue is
attributable instead of latent."*

**Measured, all three clauses are wrong.** `entity_by_name` is a `defaultdict(list)` populated with
`.append`, and `resolve_entity` returns the **first** candidate of the preferred type, so:

1. it is **payload row order that wins, not the last writer**;
2. because the `preferred_order` loop **returns early**, **no `ambiguous_entity_reference` warning is
   emitted at all**;
3. the residue is therefore **not** "attributable" — it is exactly as latent as before.

**The reviewer's decoy, reproduced:** rows `[serine (synonym "Glycine"), glycine]` with a reaction input
`"Glycine"` bind to `cmp_1 = serine`, **silently**, with nothing in the report.

**This is the same harm class C-050i exists to prevent** — a reference binding to a biologically different
row with no diagnostic — reached through the **aliases** surface rather than the `name` surface. It sits on
precisely the arm the card's own suite calls *"where a bypass would hide"*, and until now it had **no finding
ID and no owner**.

**Scope note.** C-050i's guard groups on `_norm(name)` and is correct not to fire here: the two rows are
genuinely distinct entities, so refusing would be wrong. The defect is in **`resolve_entity`'s silent
first-wins on an ambiguous alias index**, not in the dedupe.

**Owner: a new card `C-050k`. NOT C-050i** — assigned by name, not by proximity (REV-050h's lesson on
F-039). C-050i's only obligation is to **stop misdescribing it**: correct the docstring to the measured
behaviour and cite this finding.

---

## F-049 — two structural gaps the C-050i review exposed in the gate set

**Registered 2026-08-17** by REV-050i.

### 1. `tests/test_prefreeze_third_export_seam.py` is owned by no chunk

Measured: the filename occurs **zero times** in `TEST_MATRIX.md` and **zero times** in
`evidence/chunk_d_gate.py`. It is in neither Chunk D's nor SMOKE's file list.

**This is the structural reason C-050i's 83 bounded evidence records missed a live regression** that a
reviewer found in one focused run. `LEDGER.md` names this exact test **by name** as the trip-wire C-050h
withdrew a widening for, and in the same sentence hands the residue to C-050i — so the card inherited both
the residue and its known trip-wire, and still could not have caught it, because **no gate it was told to
run contains the file**.

This is the **same class** as C-053's Q9 (four of five files it must change are in no chunk) and C-052's
UNASKED-2 (five of seven). **Three cards in one pack have now hit it.** The gate set's file coverage is not
a per-card oversight; it is a hole in `TEST_MATRIX`. **Unowned — needs a card that audits which test files
belong to no chunk and closes the set.**

### 2. `tests/test_strict_failure_replay.py::…[only_unrelated_reactions_survive]` fails at base

Two parametrizations fail at `8f7514f` — **pre-existing, reproduced against base `src/`, not caused by any
card in flight**. Another IR-building file that **no chunk runs**, so the failure has been invisible.
**Unowned.** Do not let a later card discover this and mis-attribute it to its own diff.

## F-050 — the D-025 budget command measures hand-authored and generated evidence together

**Registered 2026-08-17** by C-053, which stopped at its ceiling and, in reporting the number, showed the
number is not measuring what the ceiling means.

The command quoted in every charter this sprint is:

```
git diff --numstat <base> HEAD | grep -v "evidence/g11/" | awk -F'\t' '{s+=$1+$2} END {print s}'
```

`D-025` ceiling 1 is **hand-authored additions plus deletions**, and ceilings 2 and 3 separately budget
**generated** artifacts. But this command's only exclusion is `evidence/g11/` — so **generated evidence JSON
that does not live under `evidence/g11/` is counted as hand-authored.**

**Measured on C-053:** the literal command reports **2974**, of which **1869** is generated
`evidence/c053_*.json` output. Actual hand-authored is **1105** — `src` + `tests` **722** plus
`evidence/*.py` **383**. The literal figure and the charter's own derivation were measuring different things,
by a factor of roughly 2.7.

**Consequence.** Ceiling-1 overages have been the orchestrator's error every time (REV-051a), and this is
part of why: the instrument over-reports whenever a card writes generated evidence outside `evidence/g11/`.
A card that trusted the literal number would appear catastrophically over budget and might cut real work to
"fix" it — the exact failure D-025 forbids.

**Corrected command — quote this one from now on:**

```
git diff --numstat <base> HEAD -- src tests 'docs/pwml_recovery_sprint/evidence/*.py' \
  | awk -F'\t' '{s+=$1+$2} END {print s+0}'
```

It measures exactly what ceiling 1 names: production code, tests, and hand-written evidence tooling.
Generated evidence is counted by ceilings 2 and 3, where it belongs. **Charters already dispatched carry the
old command; when ratifying an overage, re-measure with the corrected one before concluding anything.**
Unowned as a control-plane cleanup; the corrected command is usable immediately.

---

## F-054 — F-049's enumeration is closed: 119 of 147 test files belong to no chunk, and the predicate three cards used to certify coverage is unsound

**Registered 2026-08-18** by an independent read-only audit at `1c06918`. **This closes the enumeration
F-049 asked for.** F-049 stands; it was right about the shape and understated the size by an order of
magnitude.

### 1. The measured partition

`tests/` holds **155 files**, of which **147 are collectable `test_*.py` modules** (plus 2 helper modules,
1 `test_`-prefixed JSON data file, and 5 fixture JSONs).

**28 of 147 are in a chunk. 119 — 81% of the suite — are in none.**

| Chunk | Files | Source |
|---|---|---|
| A | 6 | `TEST_MATRIX.md:213` |
| B | 6 | `:214` |
| C | 8 | `:215` |
| D-core | 5 | `chunk_d_gate.py:63-67` |
| D-s8 | 1 | `:69` |
| D-qb | 1 | `:70` |
| E | 1 | `TEST_MATRIX.md:218` |

Static `def test_` across the 119 chunkless files is **1843**, and static counts run ~88% of collected node
IDs on calibrated samples (the 20 SMOKE files give 415 static against 460 collected), so roughly **2100
collected node IDs are covered by no standing gate**.

**The two membership sources agree exactly** — symmetric difference is empty for both Chunk D and SMOKE, no
file is in two chunks, and no chunk names a file that does not exist. There is **no marker, no glob, no
`conftest.py`** anywhere in the repo; membership is two hand-maintained literal lists.
`evidence/baseline_suite.py:48-53` globs all 147 for the `BASELINE.md` full-suite capture, which proves a
residual glob is cheap to write — it is simply not a merge gate.

### 2. ⚠ The certification predicate three cards used is measurably unsound

`c056b_gate_counts.json:58` and `c030a_gate_counts.json:40` certify a file as chunkless by *"the filename
occurs 0 times in `TEST_MATRIX.md` and 0 times in `chunk_d_gate.py`"* — the predicate F-049 itself states.

Measured against the real partition it produces **no false negatives and exactly one false positive:
`tests/test_map_ids.py`**. `grep -c test_map_ids TEST_MATRIX.md` returns **2**, both substring hits inside
`test_map_ids_name_gate` (Chunk C). `test_map_ids.py` itself carries **42 test functions and is in no
chunk**. A card running the published heuristic on it would have concluded, wrongly, that it was covered.

**The sound predicate is stem-exact membership** in `TEST_MATRIX.md:213-218` ∪ `:242-252` ∪
`chunk_d_gate.py:63-70`. Every future card and reviewer uses that.

### 3. Two of F-049's three worked examples were understated

* **C-053: five of five, not four of five.** As merged (`3fde1f1`) it touched exactly five test files and
  **all five are chunkless**. `LEDGER.md:620-622` names a different five and phrases
  `test_batch_driver_seam_golden` as though it were the chunked one; it is not — zero hits in both sources.
* **C-052: the claim was 5 of 7 at planning time; the merged diff (`c0df0d0`) touched 2 test files and
  both are chunkless.** The substance holds and is in fact absolute rather than 5/7.
* `tests/test_prefreeze_third_export_seam.py` — **confirmed** at zero and zero. It reaches **two** card
  seams directly (`pwml/ir.py` → C-050j, `app/streamlit_app.py` → C-055).

### 4. Minimum focused sets for the five open cards, by direct coupling

| Card | Seam | Chunkless files directly coupled | static test fns |
|---|---|---|---|
| C-050j | `pwml/ir.py` | **14** | 220 |
| C-055 | `app/streamlit_app.py`, `rag/controller.py` | **13** | 146 |
| C-056c | `release_status.py`, `strict_quarantine.py` | **7** | 74 |
| C-057 | `strict_quarantine.py` | **6** | 65 |
| C-054 | `bench/goldset.py` | **5** | 58 |

Union: **32 distinct files, ~440 static test functions (~500 collected)**. Second-order reach roughly
doubles it. **Seven of C-055's thirteen load `streamlit_app.py` by file path for AST extraction or exec** —
so a module-level script-body edit hits them hardest, and none of them is in a chunk.

### 5. Two specific exposures worth naming

* **`tests/test_measurement_tree_pin.py` is in no chunk and guards the G11 measurement-pin infrastructure
  itself** — `evidence/tree_pin.py` and `evidence/pinned_pytest.py`. A regression in the exit-98 wrong-tree
  launcher would be caught by **nothing** in the mandated gate set. That is the instrument that certifies
  every other measurement.
* **`tests/helpers_eligibility.py` has no chunked consumer at all.** Its only consumer,
  `test_paper_eligibility_corrections.py:72`, is itself chunkless, so a change to that helper is invisible
  to every chunk. (`helpers_prefreeze.py` is better off — six chunked files reach it.)

### 6. Owner and remedy

**Still unowned, and the per-card focused set is a workaround, not the fix.** The durable remedy is a
residual gate — a chunk F, or a glob over `tests/` minus the 28 — added to `TEST_MATRIX.md`, since
`baseline_suite.py` already demonstrates the glob. **Until that exists, every card names its own set by the
stem-exact predicate and runs it explicitly, and no reviewer accepts a chunk result as evidence about a
chunkless file.**

**Repaired in the same commit as this finding:** `.claude/agents/pwml-test-runner.md:52-54` said Chunk D =
177 with a core of 150. The gate says **187 = 160 + 4 + 23**. Every test-runner dispatch since C-050k merged
was reading a stale count.

---

## F-055 — the gate-failure path re-classifies from nothing and discards the boundary's computed semantic verdict

- **Severity** **HIGH** · **Registered 2026-08-18** from `runs_verify/2026-08-18_1328`, integration `ad64e86`
- **Reproduced on BOTH gate-failed legs.** Not incidental.

The same leg produces two release records that disagree. `PMC12452463/strict`:

| field | `quarantine_report.json` → `release` | manifest row → `release_status` |
|---|---|---|
| `semantic_evaluation` | **`failed`** | **`not_evaluated`** |
| `semantic_failed_checks` | **`['no_real_id_or_name_conflict']`** | `[]` |
| `completeness` | `0.5625` | `None` |
| `missing_anchors` | 7 anchors | `[]` |
| `coverage_evaluated` | `True` | `False` |
| key count | 17 | 13 |

`PMC12096016/strict` is identical in shape: `failed` → `not_evaluated`, `completeness 0.823529` → `None`,
3 missing anchors → `[]`.

**Mechanism.** `batch/driver.py` has two terminal paths. The strict PASS path — **`_finalize_pwml_export`,
defined `:1814`** — carries the boundary's record at `:1836`, `release = _frozen_release_record(pwml_result)`.
The gate-failure path (`:1770-1773`) instead **constructs a fresh classification from nothing**:

> **⚠ CORRECTION 2026-08-20.** This paragraph originally named the PASS-path function **`_finalize_strict_pass`**.
> **No such symbol exists** — `grep` over `driver.py` returns **zero** hits, verified twice. The real function
> is **`_finalize_pwml_export` (`:1814`)**; its `_frozen_release_record` call is at `:1836`, which the original
> `~:1837` cited almost correctly. Found by C-056d, confirmed by the orchestrator.
>
> **The error propagated:** it was copied verbatim into C-056d's charter OUT list, where it would have told a
> card not to edit a function that does not exist. **Harmless only because the card verified the name instead
> of trusting it.** Third charter-address defect of that session — see PACK 9 RULINGS 2 and 6.
>
> Two further addresses in this finding drifted and are corrected inline above: the `classify_release_status`
> expression is `:1770-1773` (`:1768` is the import), and `_drive`'s gate-failure `return` is `:2209`.

```python
outcome.release_status = classify_release_status(
    pipeline_executed=True, strict_gates_passed=False)
```

No semantic argument is passed, so `classify_release_status`'s defaults fire
(`release_status.py:368-369`). `_release_status_row` (`:647-667`) then faithfully serializes whatever it was
handed — its own docstring states the principle the caller breaks: *"nothing here re-classifies … because a
classification produced after the freeze is a merge rule 8 defect, not a convenience."*

**Consequence.** `acceptance.py:256` derives `runtime_semantic_refuted` from `== SEMANTIC_FAILED`. With the
manifest carrying `not_evaluated`, it is `False` even though the runtime wrote `failed` — so **C-056b's
subtractive rule never fires on any gate-failed leg.** A second inflation path, in the driver, that C-056b's
seam never sees. The scorer's own comment (`acceptance.py:744-751`) says `semantic_failed_checks` *"travels
per leg, verbatim"*; it arrives empty.

**Second symptom, one root cause.** Every gate-failed manifest carries
`"no semantic evaluation is wired into the runtime classification yet; this is a missing input (C-056a), NOT
a semantic failure and NOT a pass"` (`release_status.py:58-61`) — **C-056a merged at `93594aa`**, and the
leg's runtime recorded exactly a semantic failure. D-032 clause 6 class, same as REV-051's F-1.

- **Owner** a new card **C-056d**: `driver.py :: _finalize_gate_failure` + the `SEMANTIC_INPUT_NOT_WIRED`
  constant. Lands **after C-056c** so the restored carry includes evaluability. `_finalize_gate_failure` was
  C-041's; `LEDGER.md:437` already records its docstring as stale with no current owner — same card.
- **Not C-056c's.** `batch/driver.py` is outside its boundary and D-054 §2 records it needs zero driver edits.
- **No historical figure moves** — verified: 143 committed rows, **0** carrying a `release_status`.

## F-056 — T-102 is runnable but not green-able; two blockers nobody owns

- **Severity** MEDIUM · **Registered 2026-08-18**. **Closes the question F-009 opened on 2026-08-05.**

**F-009 CONFIRMED on both claims.** `grep -rn "taxonom" src/t2pw/sbml/` → **zero hits across all seven
modules**; the MIRIAM map (`json_to_sbml.py:760-765`) has no taxonomy. And `canonical.py:226` `_KINDS` omits
species. Traced on the committed `PMC12856317` artifacts: JSON `("9606",)`, PWML `pathway.pwml:32`
`<taxonomy-id>9606</taxonomy-id>` → `("9606",)`, SBML `()`. At `canonical.py:801` that becomes a
`Difference`, and **one `Difference` forces `verdict = not_equivalent`** (`:855`). Already pinned by a
committed passing test (`test_canonical_biological_equivalence.py:447-448`). The failing axis is exactly
`organism_context`, only on `(json, sbml)` and `(pwml, sbml)`; `(json, pwml)` passes (`:421`).

**The product owner already ruled** — `LEDGER.md:821`, PACK 2, 2026-08-13, verbatim: *"if T-102's organism
dimension is unreachable solely because SBML lacks taxonomy annotation, that exact limitation is recorded
truthfully and the other benchmark dimensions continue"*. **This is not a narrowing of D-016.** T-102's
terminal status must be **`MEASURED — organism/SBML axis structurally unreachable (F-009)`**, never `PASS`.

**Two blockers independent of F-009, both UNOWNED:**
1. **No `canonical_graph_sha256` baseline exists.** `grep -rl` over `runs/` and `runs_verify/` → 0 files, and
   `stamp_report` has **zero call sites in `pwml/writer.py`**, so the CLI re-export emits none. T-102's
   *"identical `canonical_graph_sha256`"* clause has no stored comparand and must compute both sides.
2. **Nothing drives `biological_equivalence` end to end** — its only caller in the tree is its own test file.
   An offline evidence probe (~80 lines) must be written; that is unowned work needing a grant.

**No SBML artifact has ever existed for any leg** (`build_legacy_sbml: bool = False`,
`streamlit_app.py:2636`; production passes `False` at `:6015`). The existing `outputs/pathway.sbml` is a
*clove leaves/buds* pathway — comparing PMC12856317 against it would be nonsense.

**Resolvers with NO disable switch** (matters for *"all resolvers disabled"*): pre-freeze compound resolution
(no env var, no flag); **`db_resolver=None` ENABLES the DB** rather than disabling it
(`compound_resolution.py:476-480`) — the same trap as F-051, one layer down; species canonicalization
(`prefreeze_resolution.py:1253-1276` accepts and ignores `db_resolver`/`strict_db`); enrichment
(`enrich_entities.py:1862-1867` — *"there is none"*). **`--non-strict-db` is severity-only
(`writer.py:2825`) and using it to satisfy "resolvers disabled" would be a merge-rule-6 violation.**

**Three more stale records; code wins.** `canonical.py:20-21` calls C-045 *"planning-only and undispatched"*
— it is merged **and wired** (`prefreeze_resolution.py:1410-1413`). `canonical.py:12-13` says C-052 wires the
comparator — C-052 merged and did not; it is still unwired. `DECISIONS.md:546-549`'s T-102/C-045 scheduling
blocker is **discharged**.

### ⚠ CORRECTION 2026-08-20 — blocker 1 became false, and blocker 2 sharpened. Re-measured at `32f3a57`.

**Blocker 1 as written is now FALSE, and was true when written.** F-056 stated *"No `canonical_graph_sha256`
baseline exists … `grep -rl` over `runs/` and `runs_verify/` → 0 files."* Today that grep returns **one**:

```
runs_verify/2026-08-18_1328/papers/PMC12096016/research/final_stage3_gate_report.json
  "canonical_graph_sha256": "2597ca91faea3baa0d02b066b3fc1250baa6a6a9b7714c35a515f3ea964f2335"
```

It arrived with the T-100 run committed at `8ea52c4` — **after** F-056 was registered on 2026-08-18. The
record was accurate at the time and decayed, exactly the pattern this sprint keeps paying for.

**But T-102's blocker survives on narrower and now-accurate grounds:** T-102 is specified on **PMC12856317**
(`TEST_MATRIX.md:479`), and the one baseline that exists is **PMC12096016**, **research** mode. It is the
wrong paper and the wrong mode. `stamp_report` still has **0 call sites in `pwml/writer.py`** (re-verified),
so a CLI re-export still emits none. **Restate blocker 1 as: no `canonical_graph_sha256` baseline exists for
PMC12856317.**

**Blocker 2 re-confirmed unchanged.** `biological_equivalence` is defined at `canonical.py:827` and its
**only** callers repo-wide are inside `tests/test_canonical_biological_equivalence.py`. Zero production
callers. The ~80-line offline probe remains unowned work needing a grant.

**F-009's axis re-confirmed on all four legs:** `grep -rn "taxonom" src/t2pw/sbml/` → **0 hits across 7
modules**; `canonical.py:226` `_KINDS = ("compound", "protein", "protein_complex", "nucleic_acid",
"element_collection")` — **omits species**; `canonical.py:855-856`
`verdict=(VERDICT_NOT_EQUIVALENT if diffs else VERDICT_INCOMPLETE if gaps else VERDICT_EQUIVALENT)` — **one
`Difference` forces `not_equivalent`**; and `find runs/ runs_verify/ -iname '*sbml*'` → **0 files**, so no
SBML artifact exists for any leg.

**One REFINEMENT to F-056's SBML claim.** F-056 says production passes `build_legacy_sbml=False` at
`streamlit_app.py:6015` — true. It does not mention that **`:6472` passes `True`**. Read at the committed
tip, `:6463-6472` is a manual Streamlit UI control:

```
with st.expander("Legacy SBML Export", expanded=False):
    st.caption("SBML is a legacy export path. Use PWML above for primary output.")
    if st.button("Run legacy SBML export", key="run_legacy_sbml_export_btn"):
        ... run_post_pipeline_sbml_artifacts(final_payload, build_legacy_sbml=True, ...)
```

So SBML is **not absent from the code** — it is absent from every **automated** leg, reachable only by a
human clicking a button the batch driver never clicks. **The conclusion is unchanged and the mechanism is
sharper:** the organism/SBML axis is unreachable for any benchmark leg, not because SBML cannot be built,
but because nothing in the automated path ever builds it.

Evidence: `evidence/g11/T-102/01-canonical-axis.json`, `02-canonical-axis-pythonpath.json`.
**See also F-066** — the isolated run of `tests/test_canonical_biological_equivalence.py` is what exposed the
`sys.path` defect.

## F-057 — RAG gap-filler re-imports an existing reaction under an alias enzyme name, creating the duplicate-identity orphan that refused BOTH strict legs

- **Severity** **HIGH** · `product_contract_violation` (§2, §7) · adjudicated by `pwml-bio-auditor`

**This is the single proximate cause of both strict refusals.** Both legs:
`quarantine_report.json → /refusal_reasons = ["degree_zero_export:1"]`, and
`/strict_invariants/degree_zero_exports = [{"bucket": "proteins", "name": "Isochorismatase (EntB)"}]`.

The RAG carrier attributes it precisely. `PMC12096016/strict/rag_admission_report.json → /counts =
{accepted: 2, rejected: 337}` — **both** accepted records are the *same* reaction, `enzymes:
["Isochorismatase (EntB)"]`, `source_paper.source_id: "PMC12452463"`, identical span, `reasons:
["fills_named_gap_directly: via isochorismate"]`, two different gap ids. Leg A is the same shape.

Effect: a duplicate protein `/entities/proteins/6` `Isochorismatase (EntB)` bearing **the same UniProt
`P0ADI4` as `/entities/proteins/1` EntB**, a duplicate compound, and a duplicate reaction.
`normalization_stats`: `n_entities_deduped: 0`.

**ORCHESTRATOR RE-DERIVATION 2026-08-18, independent of the adjudication.** Read directly from
`runs_verify/2026-08-18_1328/papers/PMC12096016/strict/rag_admission_report.json`:
`counts = {considered: 339, rejected: 337, accepted: 2}`, and the two accepted records are identical in
substance — same reaction `"Isochorismate to 2,3-Dihydro-2,3-Dihydroxybenzoate (DHB)"`, same
`enzymes: ["Isochorismatase (EntB)"]`, same `source_paper.source_id: "PMC12452463"`, same
`reasons: ["fills_named_gap_directly: via isochorismate"]` — **differing only in `gap_id`**:
`gap-dangling_reaction-555124de` and `gap-dangling_reaction-7e0b4a06`.

**That sharpens the affected surface.** The defect is in two places, not one: the **gap detector** minted
**two distinct gap ids for a single dangling edge**, and the **admission gate has no cross-gap dedup**, so
one span satisfying two ids is admitted twice. A card fixing only the admission gate would still leave the
detector double-counting gaps; a card fixing only the detector would leave the gate admissible to any future
duplicate-id case. **Both halves must be named in the charter.**

**The gap it "filled" was not a gap** — the paper's own `/processes/reactions/1` already covers
isochorismate → 2,3-diDHB. The detector fired twice on one dangling edge and admitted the same span twice.

**Gold warned about this exact alias.** PMC12452463 `notes`: *"The review calls EntB both 'isochorismatase'
and 'isochorismate lyase' in different sections; these are one protein, not two."*

- **Affected** `rag/admission.py` (a "gap" already covered by an existing locked reaction is not a gap);
  alias canonicalization (two protein rows sharing one accession must collapse).
- **Confidence** High on mechanism and attribution, carrier-backed on both legs. **UNVERIFIED**: why the row
  survived `_prune_entities` (name+synonym keys, `strict_quarantine.py:1325-1326`) but failed
  `_degree_zero_exports` (name-only, `:1741-1745`) — that needs the quarantine *input* payload, which is
  hashed (`admitted_payload_hash`) but **written nowhere**.

## F-058 — `EntE` fabricated as the transporter on every transport, contradicting the evidence span it cites

- **Severity** **HIGH** · `product_contract_violation` (§1 hard limit; §2)

`PMC12096016/strict/stage1_payload.json → /processes/transports/0/transporters = null` and `/1 = null`.
After merge: both read `{"protein_complex": "EntE", "provenance": "inferred", "confidence": 0.9,
"source_refs": ["secreted to the extracellular environment by a TolC-dependent process"]}` and one citing
TonB. **The evidence span names TolC and TonB; the entity asserted is EntE**, an adenylation enzyme.

Leg A is worse: Stage 1 had one transport with the correct `FepA`; the merged payload adds an `EntE`
transporter to it **and** an entirely new `enterobactin secretion` transport with `EntE`.

**Gold**: PMC12452463 `notes` — *"Export of enterobactin from the cytoplasm is never described at all, so no
efflux step may be emitted."* PMC12096016 `export_rationale` — *"Export must exclude MenD, LDH and the
transport mentions."*

- **Affected** the transporter-attachment site between the Stage-1 boundary and `merge_additions` —
  **UNVERIFIED which.** `transporters_attached: 0` rules out `process_normalizer`; no `rag_provenance`
  carrier rules out RAG import. **Instrumentation gap**: actor sub-entries (`enzymes[]`, `modifiers[]`,
  `transporters[]`) carry a bare `provenance: "inferred"` string with **no stage**, so attribution stops at
  "between Stage-1 and merge". Do not guess a stage.

## F-059 — RyhB: an sRNA sits in the `proteins` bucket, and the identity gate keys on bucket, not on declared `class`

- **Severity** **HIGH** · `product_contract_violation` (§2)

Paper: *"the RyhB sRNA (small RNA)"* — never called a protein. Stage 1 typed it
`/entities/proteins/7.class = "protein"`. **The audit corrected the class and did not move the row**:
`final_mapped.json → /entities/proteins/6.class = "rna"`, rationale *"reclassified RyhB to RNA based on
explicit evidence"*. The gate then fired **against a row it had itself labelled `rna`** —
`contract_reports.json → errors/0`: `path: /entities/proteins/6`, `detail.class: "rna"`, reason demands a
UniProt/DrugBank id.

The relocation surface exists and correctly abstained: `nucleic_acid_name_verdict`
(`process_normalizer.py:804`) is **name-shape only** and a bare sRNA symbol matches neither pattern.
`nucleic_acid_named_entities_relocated: 0`, `entity_type_mismatches_flagged: 0`.

**Gold agrees with the auditor, not the pipeline.** PMC12452463 `forbidden_identifiers[RyhB]`: *"A small RNA,
not a protein and never an enzyme."*

- **Cost** two real, paper-quoted regulatory relations lost — `RyhB inhibits EntC` and `RyhB inhibits EntF`
  quarantined as `quarantined_unmapped_entity`.
- **Affected** `process_normalizer.py` — the relocation needs a **class-declared** trigger, not just a
  name-shape one.

## F-060 — a `not_evaluated` identity outcome is rendered as `implausible_name_match` and the accession is stripped

- **Severity** **HIGH** · `product_contract_violation` (§8)

Identity resolution **did produce the correct accession**. `PMC12096016/strict/final_mapped.json →
/entities/proteins/5/mapping_meta`: `unverified_identity_claim.identifiers.uniprot = "P0A9A9"` — the exact
UniProt for E. coli Fur — with `identity_evidence = {status: "not_evaluated", lookup: "", detail:
"no_identity_evidence_source"}`, then `rejected_mapped_ids = {"uniprot": "P0A9A9"}` and `resolution.issue =
"implausible_name_match"` with `name_gate = {}`.

Three §8 problems:
1. **The ladder never ran.** `no_identity_evidence_source` is returned only when `provider is None`
   (`uniprot_evidence.py:450-462`); the source is *"off by default on purpose"*, gated on
   `T2PW_IDENTITY_EVIDENCE`. PathBank's evidence was manifestly inadequate — the rejected candidates are
   `sdhB, iscS, glpE, tusA, fdnH, fdoH, tusE, frdB`, all score 0.65, **none named Fur**.
2. **`not_evaluated` was rendered as a refutation.** `map_ids.py:5539-5571` files the claim (§8-compliant)
   then **falls through** to `_strip_rejected_identifiers` and writes `_NAME_GATE_ISSUE`. §8: `not_evaluated`
   is *"never treated as `false`"*; stripping is permitted only on `rejected`. No rung judged the name.
3. **The run configuration is not recorded** — `T2PW_IDENTITY_EVIDENCE` appears in no manifest, plan or log.
   Whether it was off by default or deliberately disabled is **UNVERIFIED**.

**Divergence:** the same protein got `P0A9A6` in the other leg. Two legs, two accessions, neither verified —
distinguishing them is exactly what the §8 ladder exists for.

**Do not materialize P0A9A9 without verification** — §2 forbids false real identifiers. The stripping is the
finding, not the drop: gold lists Fur under `acceptable_enzymes`, not `expected_enzymes`.

## F-061 — the coverage matcher false-negatives on abbreviations and complex components, and false-positives on substring collisions

- **Severity** MEDIUM · `product_contract_violation` (§4, §7) — **reporting only; non-decisive in this run**

`unmatched_terms` lists `ATP` while the admitted process carries `"adenosine triphosphate"` and the row's own
`synonyms: ["ATP"]` holds the answer. `_term_matches` (`strict_quarantine.py:825-836`) is bidirectional
substring over normalized strings and never consults synonyms or `mapping_meta.resolved_name`.

**It errs in both directions.** `MenD (competitive isochorismate-utilizing enzyme)` is reported **matched**
though MenD is absent from the payload entirely — because the normalized term *contains* `isochorismate`. So
the report simultaneously declares a present participant missing and a gold-**forbidden** entity covered.

Leg A is understated far more: its unmatched `EntC, EntB, EntD, EntF` are all present with accessions inside
PathBank complex wrappers, and `_process_core_terms` (`:806-822`) collects the complex **name** but not its
**components**. Leg A's published `completeness = 0.5625` sits against a `min_core_coverage` of `0.5` — one
more name substitution from a **false** `requested_core_coverage_below_minimum`.

**Fix the matcher, not the gold set.** Gold's ATP entry is quote-backed and correct. **The threshold value
does not move** (§7).

## F-062 — structural refusal reasons bypass C-041a's D-002 seam, making `review_required` unreachable after any quarantine refusal

- **Severity** **HIGH** · `product_contract_violation` (§4, §7, **§13**, merge rule 7)
- **⚠ THIS CONTRADICTS A LOCKED CONTRACT POSITION.**

`strict_quarantine.py:2025-2034`: coverage reasons are converted to `review_reasons` when `defensible_core`
holds, but `structural_reasons` are appended to `refusal_reasons` **unconditionally**. Both legs had
`defensible_core = True` (`minimum_core_satisfied: true`, `coverage.reasons: []`, `core_accepted` 6 and 9)
and `coverage_reasons = []` — yet `ok = false`, which drives `release_status.py:417-419`
`elif not strict_gates_passed: status = DIAGNOSTIC_ONLY`.

A degree-zero orphan is a **removable row**, not an unsupportable biological requirement — the pruner already
owns that reason code (`degree_zero_after_quarantine`) and applied it to Fur, EntF, RyhB and
`enterobactin synthase` in the same run.

**`PRODUCT_CONTRACT.md:341` (§13, LOCKED), verified verbatim:** *"PMC12452463 — … Correct outcome **after the
index fix** is `review_required` with `strict_acceptance_eligible=false`. **Never strict success.**"*

The run produced `diagnostic_only` with `strict_acceptance_eligible: false`. **The eligibility flag matches
the contract; the status does not.**

**⚠ ONE QUALIFICATION, AND IT IS THE ORCHESTRATOR'S CORRECTION TO THE ADJUDICATION.** The clause is
conditioned on *"after the index fix"*, and **that phrase occurs exactly once in the entire control plane,
with no antecedent anywhere** — `grep -rn "index fix" docs/pwml_recovery_sprint/*.md` returns that single
line and nothing else. So whether the condition is satisfied is **UNVERIFIED**, and this finding must **not**
be quoted as a settled contradiction of a locked position until the referent is named.

**What is verified and does not depend on the referent:** the mechanism below is read directly from the code
and reproduces on both legs — `review_required` is unreachable after any quarantine refusal, however
defensible the core. That stands on its own.

**The undefined referent is itself a control-plane defect** and is registered here: a locked contract row
conditions a required outcome on an event the control plane never defines, so no agent can determine whether
the row is currently binding.

- **⚠ BLOCKING SEQUENCING — merge rule 6.** **F-062 must NOT be fixed before F-057 and F-058.** Repairing the
  refusal seam in isolation would ship leg B's fabricated `EntE` transporters and its LDH-derived NAD+/NADH.
  The duplicate-identity orphan and the transporter fabrication land first.

## F-063 — Stage-1 emits a dangling interaction endpoint, and the audit "resolves" it by deleting the relation

- **Severity** MEDIUM · `product_contract_violation` (§2) · **owner is Stage 1, NOT C-060**

`stage1_payload.json → /processes/interactions/1/entity_2 = "ent gene clusters"` with no matching row in any
`entities` bucket — a **dangling reference**, not a hallucination: the phrase is verbatim in the source.

**C-060's gate WAS reached and correctly abstained.** `merged_payload.json → /entity_admission_report` is
present (`{removed: 0, demoted: 4, admitted: 0}`), so `screen_additions` ran. It must not fire: the span *is*
locatable, and `entity_admission.py:44-46` confines removals to `entities.compounds` and `processes.*` under
two rules, neither of which addresses an undeclared participant. **C-060 is not the owning surface.**

The vocabulary that recognises the string already exists — `process_normalizer.py:242-245`
`NUCLEIC_ACID_TAIL_RE` matches `gene\s+clusters?` — but is applied only to existing protein-bucket rows,
never to unregistered process participants.

**Instead the audit deleted the biology**: *"Removed the invalid interaction that referenced an undefined
entity 'ent gene clusters'"*. That deletion is the sole reason Fur ends up degree-zero and pruned. The
relation is gold-acceptable and paper-explicit.

- **Affected** `stage_one_boundary.py` — every process participant is a declared entity, or the process does
  not cross the boundary.

## F-064 — the ID-mapping cache is saved by a non-atomic unlocked whole-file overwrite, and a failure there destroys a completed leg

- **Severity** MEDIUM · **Registered 2026-08-18**

`mapping/map_ids.py:780-783`:

```python
def save(self) -> None:
    if not self.enabled:
        return
    self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
```

A truncate-and-rewrite of a **4.4 MB shared cache** — no temp file, no `os.replace`, no lock, no error
containment. `PMC12452463/research` died at `post_pipeline` after **456.8 s** with `[Errno 22] Invalid
argument` on that path, **after producing 5 reactions and 20 entities**.

**Attribution: NOT a reproducible code defect.** The discriminating experiment came back negative —
`PMC12096016/research`, same mode and same code, **PASSED**. Three `git worktree add` invocations by the
orchestrator ran during the run, each writing ~4,230 files, one at 13:50 inside leg 2's window. **Recorded as
an orchestrator-induced infrastructure failure.** Do not cite leg 2 as evidence about the pipeline.

**The fragility is still real and independent of what triggered it**: research mode is fail-open by design,
so a cache write must not be able to fail a leg that has already completed its science. No committed run has
ever recorded an `Errno 22`.

**Standing operational rule adopted:** no worktree creation or heavy filesystem work in the primary checkout
while a pipeline leg is running.

## Instrumentation gaps blocking firmer attribution (all three UNOWNED)

1. **The quarantine input payload is not persisted.** `admitted_payload_hash` names a payload written to no
   file, so F-057's survival/condemnation asymmetry cannot be closed.
2. **Actor sub-entries carry no lineage** — `enzymes[]`, `modifiers[]`, `transporters[]` carry a bare
   `provenance` string with no stage. This is why F-058 cannot name a stage.
3. **Run configuration is not recorded** — `T2PW_IDENTITY_EVIDENCE` appears in no manifest, plan or log.

Also noted, non-blocking: `cofactor_policy.py:3-4` still says *"UNWIRED BY DESIGN: nothing calls this"* —
stale, `entity_admission.py:66` imports it. And leg B ran the audit at `temperature = 0.14` (leg A: `0.0`) —
a reproducibility hazard worth a product-owner note; no contract clause found that it breaches.

---

## F-065 — the `.env`-dependent red family has a SIXTH member, it sits in Chunk D core, and the standing register understates the core count

**Registered 2026-08-19.** Found independently by **two cards in the same session** — C-054 by attribution during
its Chunk D sweep, and C-056c by refusing to assume a pre-charge covered an unlisted failure. Verified
directly by the orchestrator.

### The test, and its own false premise

`tests/test_pwml_writer.py :: test_cli_export_emits_the_canonical_organism_and_keeps_its_provenance`
(def at `:1791`):

```
:1829    # P4-01: no worktree carries a .env, so ``PathBankDbResolver.from_env()``...
:1833    assert report["db_resolution"]["available"] is False
```

**The comment states the premise the assertion rests on, and that premise is now false.** With `.env`
present, `PathBankDbResolver.from_env()` succeeds, `available` is `True`, and the assertion fails as
`assert True is False`. The mechanism is the one F-051 already named at
`compound_resolution.py:476-480` — `db_resolver=None` means *"resolve from `.env`"*, not *"no DB"*.

**Proven environment-dependent, not diff-dependent**, two ways independently:
* C-054's one-file differential — `1 passed` at **both** base and tip production without `.env`.
* C-056c reproduced it **identically at base `1cbfa01`** with `.env` present, same line, same message.

### Why it matters more than the other five

The standing register (shared execution block §7, and the pre-charge lists carried in
`c056b_gate_counts.json` and `c030a_gate_counts.json`) enumerates the *"asserts the DB is NOT configured"*
class as **five** tests — four in `tests/test_prefreeze_third_export_seam.py` and one in
`tests/test_prefreeze_species_resolution.py`. **All five are chunkless**, so they are invisible to every
standing gate and cost nothing in a gate count.

**This sixth is inside Chunk D core** (`chunk_d_gate.py:63-67` lists `test_pwml_writer.py` among the five
CORE files). So it changes a mandated number:

> **Chunk D core is 159/160, and the total is 185/187, on any worktree carrying `.env`.**

Both C-054 and C-056c measured exactly that — `SETS_EQUAL=True`, 187 node IDs, `core=159/160`, `s8=4/4`,
`qb=22/23`, the two failures being this test and `qb` node15.

### The orchestrator caused the exposure, and the instruction was still right

`.env` is in these worktrees because **I instructed every card to copy it in for F-051 control parity.**
That is the correct standing remedy — *hold the tree and its `.env` constant and swap only the file under
test* — and copying it as a control on both sides is explicitly sound. The consequence is simply that a test
asserting the absence of `.env` now fails, and **the register failed to anticipate a member of that family
inside a chunked file.**

**Do not "fix" the test to make Chunk D green.** It is C-045b's own guard, and `REV-051` already recorded
that re-pointing it would make the following `preflight` assertion pass in **both** configurations,
destroying the property it exists to protect — the reviewer there endorsed the refusal to re-point it. The
same reasoning binds here: this is a **pre-charge to record**, not a defect to repair.

### Binding on every future card

1. **Expect Chunk D 185/187 with `core=159/160`** on any `.env`-carrying tree, and pre-charge this node ID by
   name alongside `qb` node15.
2. **A third failure is the card's own** and must be named.
3. Still reproduce both at your own base before claiming them (F-051).
4. The *"five tests"* phrasing in the shared execution block and in the two committed `*_gate_counts.json`
   pre-charge lists is **superseded by this finding**. It is six, and one of them is chunked.

### Related, and not fixed here

`test_pwml_writer.py:1829`'s comment is a **measurably false committed sentence** of the same class C-051d
was chartered to correct. It is not in any current card's seam. **Register, do not fix** — it needs an owner.

## F-066 — 21 test files cannot be collected in isolation, and the failure reads as a broken environment

- **Severity** **HIGH** (process, not product) · **Registered 2026-08-20**, integration `32f3a57` · orchestrator-measured
- **Discovered while running T-102's canonical-equivalence evidence.** Not hypothetical: it was hit on the
  first isolated run of the first file tried.

### The mechanism, verified four ways

There is **no `conftest.py` anywhere in the repository** (`ls conftest.py tests/conftest.py` — neither
exists), **`t2pw` is not installed into the venv** (`.venv/Scripts/python.exe -c "import t2pw"` →
`ModuleNotFoundError`; no `t2pw*` and no `.pth` in `.venv/Lib/site-packages/`), and **`pytest.ini` carries
only `testpaths` and `norecursedirs`** — no `pythonpath`.

Instead, **each test file inserts `src` into `sys.path` itself**, e.g. `tests/test_reference_repair.py:57-58`:

```python
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
```

**21 of 148 test files import `t2pw` and do NOT perform that insert.** They import the package directly at
module scope — e.g. `tests/test_canonical_biological_equivalence.py:21` `from t2pw.pipeline import canonical`
— so they collect **only** when pytest happens to import a path-inserting file first in the same process.

Measured, at `32f3a57`, same interpreter and same tree:

| invocation | result |
|---|---|
| `pytest -q tests/test_canonical_biological_equivalence.py` | **`ModuleNotFoundError: No module named 't2pw'`**, `Interrupted: 1 error during collection`, exit 2 |
| `PYTHONPATH=src pytest -q tests/…same file…` | **`29 passed in 0.19s`**, exit 0 |

G11 reports: `evidence/g11/T-102/01-canonical-axis.json` (the failure) and
`02-canonical-axis-pythonpath.json` (the pass). Both retained — the failing one is genuine evidence and is
not deleted.

### The 21 files

`test_attempt_cap_termination_reason` · `test_baseline_regression_2026_07_28` ·
`test_canonical_biological_equivalence` · `test_cofactor_policy` · `test_completeness_audit` ·
`test_entity_admission` · `test_entity_identity_contracts` · `test_failure_detail` ·
`test_locked_noop_quarantine_policy` · `test_pathbank_unknown_fallback` · `test_payload_models` ·
`test_pipeline_lineage_schema` · `test_prefreeze_compound_resolution` · `test_prefreeze_species_resolution` ·
`test_protein_export_policy` · `test_rag_graph_delta` · `test_rag_loop_controller` · `test_rag_loop_policy` ·
`test_reaction_lock_manifest` · `test_stage2_mapping_boundary` · `test_streamlit_stage8_export_contract`

### Why it has stayed invisible, and why it stops being invisible NOW

**SMOKE and every chunk gate are unaffected** — each runs a multi-file selection that always contains a
path-inserting file, so the import succeeds by accident of ordering. `test_completeness_audit.py` is in
SMOKE and is on this list; it has never failed there.

**F-054 is what makes this bite.** 119 of 147 test files are in no chunk, so every card is now instructed to
**name its chunkless files and run them explicitly** — which is exactly the isolated selection that breaks.
The two cards live at the time of writing were both affected:

* **C-055** — 4 of the 6 chunkless files its charter names: `test_rag_loop_controller`,
  `test_rag_loop_policy`, `test_rag_graph_delta`, `test_attempt_cap_termination_reason`.
* **C-050j** — 4 of the files its § 6 names: `test_protein_export_policy` (45),
  `test_prefreeze_compound_resolution` (50), `test_pathbank_unknown_fallback` (15),
  `test_prefreeze_species_resolution` (12).

Both were notified with the remedy before they reached their focused runs.

### ⚠ The reason this is HIGH and not a nuisance

The failure surfaces as `ModuleNotFoundError: No module named 't2pw'` at collection — **indistinguishable, to
an agent that has not measured it, from the `httpx` under-declaration (F-067) that really was an environment
defect.** The tempting responses are all wrong and two of them are destructive:

* pip-installing something (masks the real cause, mutates a shared venv);
* adding a `sys.path` insert to a file outside the card's boundary — and
  `tests/test_prefreeze_species_resolution.py` is **zero lines** for C-050j under C-045/D-016;
* creating a `conftest.py`, which changes collection for **all 148 files** at once;
* concluding the venv is broken and rebuilding it — which is what destroyed `.venv` once already.

**Standing remedy until this is owned and fixed: `PYTHONPATH=src` in the bounded child**, for any selection
containing one of the 21. It is exactly equivalent to the insert the other 127 files perform on themselves,
it changes no file, and it must be **disclosed in the card's report** as an orchestrator-directed invocation
change.

### Owner and remedy — UNOWNED, and deliberately not fixed here

The durable fix is one of: a repo-root `conftest.py`, `pythonpath = src` in `pytest.ini`, or installing the
package. **All three change collection semantics for every test file in the repository**, which is a
sprint-wide gate change and needs its own card and its own baseline re-measure — SMOKE and all four chunks
would have to be re-pinned. **It must not be absorbed by a card that merely tripped over it.**

Registered, not fixed. The 21-file list above is the exposure.

## F-067 — `httpx` is undeclared in both dependency files, and an unpinned rebuild silently drops it

- **Severity** MEDIUM (process) · **Registered 2026-08-20** · previously recorded only in a session handoff

`httpx` appears in **neither `requirements.txt` nor `pyproject.toml`**. It arrived transitively via `openai`;
the current `openai` requires **`httpx2`** instead, so a fresh unpinned rebuild drops it and **five SMOKE
files fail collection** with `ModuleNotFoundError`. It was installed manually when `.venv` was rebuilt on
2026-08-19 and is present today (`httpx 0.28.1`, Python 3.13.6, pytest 9.0.3).

`requirements.txt` carries **19 dependencies with zero pinned versions**.

**Binding on every environment rebuild: install `httpx` explicitly, then drift-check SMOKE against the pinned
baseline before trusting any gate.** That drift check is what caught this before it could masquerade as a
card regression.

**Do not confuse this with F-066.** Both present as `ModuleNotFoundError` at collection. This one is a real
missing distribution; F-066 is a path defect where the module is present and importable. The discriminator:
`python -c "import httpx"` versus `PYTHONPATH=src python -c "import t2pw"`.

Registered, unowned, not fixed.

## ⚠ CORRECTIONS to F-057 and F-058 — 2026-08-20, measured at `ee266ce`

Established by read-only measurement **and byte-exact replay of production functions against the committed
`runs_verify/2026-08-18_1328` artifacts**. Both findings' *conclusions* survive. F-057's stated **mechanism
does not**, and F-058 is **no longer UNVERIFIED**. This is the fifth time this sprint that a record's cited
mechanism proved false while its conclusion held.

### F-057 half (i) is **FALSE as written**. The gap detector is not defective.

F-057 states *"the **gap detector** minted **two distinct gap ids for a single dangling edge**."* It did not.

`make_gap_id` (`rag/retrieve.py:200-213`) is `sha1(f"{kind}|{label}")[:8]`, casefolded, and nothing else.
Both suffixes were recovered by replaying it:

| gap id | recovered `label` |
|---|---|
| `gap-dangling_reaction-555124de` | **`EntC reaction`** |
| `gap-dangling_reaction-7e0b4a06` | **`EntB isochorismatase reaction`** |

**Two ids exist because two ADJACENT REACTIONS were each independently dangling, on different open
metabolites** — `EntC reaction` on unfed substrate `chorismate`, `EntB isochorismatase reaction` on terminal
product `pyruvate` (`retrieve.py:958-964`, `:990-1003`). Neither is a cofactor, so `_is_open`
(`retrieve.py:945-947`) returns `True` for both. Replaying `_connectivity_gaps` (`retrieve.py:910-1004`) on
the committed `PMC12096016/strict/stage1_payload.json` reproduces **all four** `dangling_reaction` ids
exactly.

**The per-edge dedup demonstrably WORKS.** `detect_gaps._add` (`retrieve.py:706-711`) drops a repeat
`gap.key()`. Proof it fires: `EntF enterobactin synthase reaction` is dangling on **two** open ends
(terminal `enterobactin`, unfed `L-serine`), `_connectivity_gaps` emits it **twice**, and `by_gap` carries
**one** id — `gap-dangling_reaction-8b584237`.

**The actual mechanism, and the one a charter must name:** `Gap.target_names()`
(`rag/retrieve.py:316-347`) falls through to `target_symbols`, which `detect_gaps` fills from
`_reaction_participants(row)` (`retrieve.py:548-563`, set at `:749` and `:1001`). Replayed:

```
gap-dangling_reaction-555124de 'EntC reaction'                 -> ['chorismate', 'isochorismate']
gap-dangling_reaction-7e0b4a06 'EntB isochorismatase reaction' -> ['isochorismate', '2,3-dihydro-2,3-dihydroxybenzoate', 'pyruvate']
```

`isochorismate` is a participant of **both** because it is the shared metabolite of two adjacent reactions.
One candidate therefore satisfies both gaps' target tokens and is accepted twice, `fills_named_gap_directly:
via isochorismate` — exactly the `reasons` string on both accepted records.

**This is a structural property of every linear pathway: each internal metabolite lies in two adjacent gaps'
target sets.** A card that "fixes the detector" would be fixing correct code.

### F-057 half (ii) — confirmed, and narrowed

**Confirmed: there is no cross-gap dedup.** `admit_candidates` (`rag/admission.py:3115`) partitions **by gap
id** (`:3226`) and loops the ids independently (`:3229-3242`); `_admit_for_gap` (`:3253-3390`) allocates a
fresh `accepted`/`frontier`/`reachable` per call (`:3292-3296`). **No set of already-admitted claim
identities is shared across iterations, and no key on `(reaction, enzymes, span)` exists.**

The relevant keys all *include* the gap id, which is what keeps the two apart:
* `RagReactionCandidate.merge_key` (`admission.py:1917-1918`) = `(self.gap_id, self.claim_identity())` —
  and it has **zero call sites repo-wide**.
* `synthesize._dedupe_candidates` (`synthesize.py:1792-1824`) keys on
  `(candidate.gap_id, claim_identity(), provenance_identity())` at `:1815-1819`. **Both accepted records are
  identical on the last two** — same chunk `262953bcb3a53769e2cd36e4ba0a3c35`, same span. **Only
  `gap_id` at `synthesize.py:1816` separates them.**

The one rule keyed *without* gap id — `graph_delta.RULE_DUPLICATE_CLAIM`
(`rag/graph_delta.py:373-375`) — is an **inter-round** check whose only production caller is
`controller.py:297` inside `run_rag_loop`. `maybe_run_rag` calls `synthesize_with_report` directly
(`streamlit_app.py:682`), so **it never ran on this leg**. (C-055 wires the controller; that changes what
runs in round 2+, never within a round.)

**NARROWED: "admitted twice" is true of the REPORT and false of the PAYLOAD.** The two candidates collapse
to **one** row downstream at `synthesize._resolve_reactions` (`synthesize.py:1058-1121`), which groups by
`conflict_key`. `merged_payload.json` carries **one** duplicate reaction, `/processes/reactions/5`, not two.
**The harm is the row existing at all, not existing twice.**

### F-057 A-3 — the "gap" was already covered, and NOTHING checks for that

**No already-covered check exists anywhere.** Established four ways:

1. The complete rejection vocabulary (`admission.py:136-174`, `REASON_NO_GAP_ID` … `REASON_CONFLICTING_RESOLUTION`) contains **no reason code** for "already present in the seed graph".
2. The seed graph is used **only to admit**: `_pathway_metabolites(seed_payload, token)` (`admission.py:2228`, consumed `:3166`, `:3265`) makes seed metabolites **anchors that widen the frontier**. It is an admission enabler, never a refusal.
3. **No lock-manifest coupling.** `rag/admission.py` imports exactly one non-stdlib non-rag module — `from t2pw.pipeline.lineage import LineageEntry, LineageSource` (`:57`). Neither it nor `rag/retrieve.py` imports `reaction_lock_manifest` or `reaction_preservation_validator`. Lock reconciliation is post-hoc at `strict_quarantine.py:1939`.
4. The nearest thing, `synthesize._payload_closes_gap` (`synthesize.py:2957-3068`, `GAP_DANGLING_REACTION` branch `:2976-2984`), runs **after** synthesis for the unfilled-gap report. It does not gate admission.

**Why the row was not caught as a duplicate of the locked seed reaction:** seed `/processes/reactions/1` is
`isochorismate -> {2,3-dihydro-2,3-dihydroxybenzoate, pyruvate}`; the RAG row is `isochorismate ->
{2,3-dihydro-2,3-dihydroxybenzoate (DHB)}`. **Different output name, missing `pyruvate` ⇒ different
`conflict_key`** ⇒ `_resolve_reactions` (`synthesize.py:1079-1095`) never grouped them. Both survive, and
`quarantine_report.json → /admissions` shows all six reactions `core_accepted`.

### F-057 A-4 — no accession-keyed collapse exists, and `n_entities_deduped: 0` is NOT an anomaly

**Artifact fact re-read and CONFIRMED:** in `PMC12096016/strict/final_mapped.json`, `/entities/proteins/1`
(`EntB`) and `/entities/proteins/6` (`Isochorismatase (EntB)`) carry **byte-identical**
`mapped_ids = {"uniprot": "P0ADI4", "pathbank_protein_id": "6224", "gene_name": "entB"}`.

**Every dedup in the tree keys on a NAME:** `process_normalizer._dedupe_named_rows` (`:706-721`, counter
bumped `:2736` inside `canonicalize_same_as_aliases` `:2332`, wired `:5445`) · `ir._dedupe_named_rows`
(`:551`, keying `:596-599`) · `strict_quarantine._prune_entities` (name ∪ synonyms, `:1325-1326`) ·
`_degree_zero_exports` (name only, `:1741-1743`). The only accession-keyed structures are **component
lookups that silently last-wins overwrite** — `ir.py:1369`/`:1379-1383`, `writer.py:1171`/`:1203`,
`process_normalizer.py:4456`, `map_ids.py:6074-6089`.

**So `n_entities_deduped: 0` is exactly what the code predicts:** different name norms, no `same_as` link,
no pass could have collapsed them. It is not evidence of a broken deduper.

### ⚠ F-057's UNVERIFIED prune/degree-zero asymmetry — RESOLVED, and it is NOT the cause

F-057 flags as UNVERIFIED *"why the row survived `_prune_entities` (name+synonym) but failed
`_degree_zero_exports` (name-only)"*. **Both cited ranges are accurate against the code.** But **neither
protein row carries a `synonyms` key at all** — the full key list on `/6` is `mapped_ids, mapping_meta, name,
organism, pathbank_protein_id, pathbank_species_id, provenance_lineage, rag_confidence, rag_provenance,
source_papers, source_refs, species, species_id, species_name, species_ref, taxonomy_id`. With no synonyms,
`_entity_name_norms` (`process_normalizer.py:626-636`) degenerates to name-only and **the two predicates are
identical on these rows.**

**A THIRD, previously unrecorded mechanism is the actual cause.** The reference that kept the row alive was
destroyed by identifier mapping. In `merged_payload.json`, `/processes/reactions/5/enzymes[0]` reads
`{"protein": "Isochorismatase (EntB)", ...}`; in `final_mapped.json` the same actor reads
`{"entity": "isochorismatase", "entity_type": "protein_complex", "role": "catalyst", ...}`. **The enzyme was
renamed and retyped, so no surviving process references the protein row named `Isochorismatase (EntB)`** —
hence `degree_zero_exports = [{"bucket": "proteins", "name": "Isochorismatase (EntB)"}]` and
`refusal_reasons = ["degree_zero_export:1"]`.

The rewrite site is `map_ids._rewrite_reaction_protein_enzymes_to_complexes` (referenced at
`tests/test_rag_payload_gate_guardrails.py:66`). **Its exact line range is UNVERIFIED and it is outside both
cards' boundaries.** Registered here; **do not fix it under F-057.**

**Still genuinely UNVERIFIED, as F-057 says:** the quarantine *input* payload is written nowhere (only hashed
as `admitted_payload_hash`), so the state between `final_mapped.json` and the degree-zero verdict cannot be
read.

---

### F-058 — the site is PINNED, byte-exactly reproduced, and the record's scope was off by one

F-058 records the affected site as *"the transporter-attachment site **between** the Stage-1 boundary and
`merge_additions` — **UNVERIFIED which**."* **It is now verified, and it is INSIDE `merge_additions`.**

**The site:** `pipeline/pipeline.py :: _inject_name_based_modifiers` (defined `:2713`), **transports branch
`:2880-2914`**, appending at `:2907-2914`. Called at **`pipeline.py:1218`**, immediately after the additions
merge (`:1192-1216`) and before `screen_additions` (`:1219`).

**Byte-exact replay.** Running `_inject_name_based_modifiers` on a deep copy of each committed
`stage1_payload.json`, with no Stage-2 additions and no RAG, reproduces the committed `merged_payload.json`
transporter entries **character for character on all three affected legs** — `PMC12096016/strict` transports
0 and 1, `PMC12096016/research` transport 0, and `PMC12452463` strict + research transport 0 (the correct
`FepA` entry **plus** an appended `EntE` entry).

**The cause, one statement** (`pipeline.py:2884-2886`):

```python
for transport, tname, tevidence_text in transport_rows:
    tevidence = tevidence_text.lower()
    if pname_lower not in tname and pname_lower not in tevidence:
        continue
```

**A bare substring test — no word boundary, no cue window, no exactly-one-candidate guard.** `pname_lower =
"ente"` is a substring of `"ent`**erobactin**` export"`, `"ferric ent`**erobactin**` import"` and
`"ent`**erobactin**` secretion"`. **The guards the REACTION branch already has —  the cue window at
`:2840-2842` and the exactly-one-candidate refusal at `:2853-2863` — were never extended to the transport
branch.**

**A second defect on the same statement:** `:2909` writes the key `"protein_complex"` unconditionally, though
`EntE` was collected as a **protein** (`:2744-2746`) and lives in `entities.proteins` — which is why
`final_mapped.json` shows mapping rewriting it back to `"entity_type": "protein"`.

**This also explains F-058's own "Leg A is worse" observation:** on `PMC12452463` the new
`enterobactin secretion` transport row comes from Stage-2 additions via `_extend_unique`
(`pipeline.py:1202`), and `_inject_name_based_modifiers` at `:1218` then attaches `EntE` to it **in the same
call**.

**Eight other candidate sites were enumerated and EXCLUDED**, each with a cited basis — most usefully:
`process_normalizer.attach_transporters_from_evidence` (`:3290`, append `:3390`, counter `:3392`) is excluded
three ways (the counter is bumped on the same statement as the append, so `transporters_attached: 0` ⇔ zero
appends; it runs post-pipeline at `:5437-5438`, outside the window; and its row shape lacks
`provenance`/`confidence`/`source_refs`); and **RAG import is excluded structurally — `grep -c transport
src/t2pw/rag/conform.py` → 0**, so the conform envelope cannot carry a transport row at all. That is a
stronger exclusion than F-058's original "no `rag_provenance` carrier" argument.

### F-058's instrumentation gap — CONFIRMED at the type level, consequence no longer holds

Actor sub-entries genuinely carry a bare provenance string with **no stage**: `schema.py:340-358`
`PayloadReactionActor` has no `stage` and no `provenance_lineage` field (`PayloadProvenance` is a 4-value
`Literal` at `:30`); `payload_models.py:313-324` `ActorModel` has `provenance: str | None = None` at `:323`.
Every construction site writes the bare string — `pipeline.py:2866-2878`, `:2908-2914`,
`process_normalizer.py:3525-3534`, `:3676-3683`, `:3390`. The structured `provenance_lineage` carrier exists
on **rows** but on **no actor sub-entry** in any leg.

**But F-058's consequence — *"attribution stops at 'between Stage-1 and merge'"* — is now superseded.** It was
narrowed by **replay**, not by instrumentation: the writer's field set (`protein_complex` + `confidence: 0.9`
+ `provenance: "inferred"` + `source_refs == [evidence]`) is a **unique fingerprint of
`pipeline.py:2908-2914`**, emitted by no other site in the tree.

### Ownership consequence — the two cards are DISJOINT at source, NOT at test

**Source: intersection is empty.** F-057 = `{rag/retrieve.py, rag/admission.py, rag/synthesize.py}`;
F-058 = `{pipeline/pipeline.py}`. The two touch points are one-directional and need no edit on the other
side.

**Tests: three files are in both sets** — `test_rag_gap_admission.py` (18), `test_rag_payload_gate_guardrails.py`
(16) and `test_rag_seed_entity_reimport.py` (14). Whichever card lands second must re-run all three.

**Two collisions the charters must AVOID, both live:**
* an accession-keyed collapse placed in `ir._dedupe_named_rows` (`:551`) **collides with C-050j**;
* any change to `strict_quarantine._prune_entities` (`:1294`) **collides with C-057**.

**Per A-4 above, F-057 needs neither** — the prune predicates are not the cause. Both are excluded from its
boundary.

**Two untested seams, named so no one assumes coverage:** `_admit_for_gap` and `_dedupe_candidates` have
**zero direct references anywhere in `tests/`**; and **nothing asserts on the transport branch of
`_inject_name_based_modifiers` at all** — all three of its name-heuristic tests
(`test_rag_payload_gate_guardrails.py:330`, `:402`, `:428`) are on the **reaction** branch.

### Three more F-054 traps, found in these sets

1. **`test_rag_admission_adversarial` (54 tests) is NOT Chunk C** — only `test_rag_admission_production_path`
   is. A substring predicate on `test_rag_admission` pulls a 54-test file into a chunk documented at 109.
   The largest mis-certification available in this set.
2. **`test_entity_admission` (23) is NOT Chunk C** — a substring on `admission` matches it against Chunk C's
   `test_rag_gap_admission`.
3. **`test_rag_provenance_gates` (Chunk C) vs `test_rag_payload_gate_guardrails` (no chunk)** — both match a
   `gate` substring; only the first is in C. Likewise `test_pipeline_reaction_rag_provenance` (Chunk C) vs
   `test_pipeline_lineage_*` (no chunk).

## F-068 — the committed leg corpus is 35, not 32, and every census figure in the sprint cites 32

- **Severity** MEDIUM (record integrity) · **Registered 2026-08-20**, integration `f6e856b`
- Surfaced by C-050j's census, then **independently re-measured by the orchestrator**.

### The measurement

`find runs runs_verify -name final_mapped.json` → **35**, split **`runs/` 14 + `runs_verify/` 21**, which
reproduces C-050j's reported split exactly. By run directory:

| run | legs |
|---|---|
| `runs/2026-07-27_1623` | 1 |
| `runs/2026-07-28_0919` | 2 |
| `runs/2026-08-02_2130` | 11 |
| `runs_verify/2026-08-04_1148` · `_1207` · `_1358` · `_1504` · `_1647` | 1 each |
| `runs_verify/2026-08-04_1234` · `_1306` | 2 each |
| `runs_verify/2026-08-04_1754` | 9 |
| **`runs_verify/2026-08-18_1328`** | **3** |

### The delta is fully accounted for, and it is exactly the T-100 run

**32 + 3 = 35.** The T-100 Wave B run (`runs_verify/2026-08-18_1328`, committed `8ea52c4`) authorized **four**
legs and contributes **three** `final_mapped.json` files — the fourth is F-064's casualty,
`PMC12452463/research`, which died on the unlocked 4.4 MB `id_mapping_cache.json` write at
`stage: post_pipeline` after 456.8 s and never reached a mapped payload.

**So every "32" record was true when written and went stale on 2026-08-18.** Same class as F-056's blocker 1
and the seven `460` records: no record was wrong, the corpus moved.

### What this makes stale — and the one consequence that actually matters

Records citing 32 include **F-014** (*"all 32 committed `final_mapped.json`"*), **C-050i's census**,
**REV-050i's independent 32×9 re-census**, and **D-050**, which carries the *"zero of 32"* figure into
C-050j's own charter.

**The consequence that matters: C-050i's zero-of-32 census no longer covers the corpus.** Three legs — all
three surviving T-100 legs — postdate it and were never censused by it. Anyone re-quoting *"payload-authored
exposure is zero of 32"* is quoting a census that is **three legs short of the committed corpus**.

**C-050j's census is therefore strictly stronger and supersedes it for this question:** it covers all **35**
legs across **both** production `strict_db` arms — 70 measurements — and returns zero on the full corpus.

### Binding on every future census

**Measure the corpus, never cite its size from a record.** The command is
`find runs runs_verify -name final_mapped.json | wc -l`. It has changed once mid-sprint and will change again
the moment T-103 or T-104 runs — **T-104 alone would add up to 20 legs and make every figure here stale in a
single night.** A census that reports a total without having counted it is reporting a record, not a
measurement.

Registered. **The historical censuses are NOT invalidated as evidence of what they measured** — they are
accurate over the corpus that existed. Only their *coverage claim* over "the committed corpus" has decayed.
No accepted card is reopened.

## F-069 — three committed legs are outside the export golden, two tripwires fired, and nothing was listening

- **Severity** **HIGH** · **Registered 2026-08-20**, integration `cbeaa84`
- Surfaced by **REV-050j** (finding 1), **verified independently by the orchestrator**, and it corrects a
  misclassification the C-050j writer made in good faith.

### Two committed tests fail UNCONDITIONALLY on the integration branch

| test | assertion | why it fails |
|---|---|---|
| `tests/test_c030_canonical_identity_fallback.py:176` `test_the_census_reproduces_over_the_committed_corpus` | `assert len(_corpus()) == 32` | the corpus is **35** (F-068) → `AssertionError: assert 35 == 32` |
| `tests/test_compound_resolution_extraction.py:806` `test_the_golden_covers_every_committed_leg_fixture` | every `final_mapped.json` under `runs`/`runs_verify` must appear in `GOLDEN` | **three** legs are missing from `GOLDEN` |

The three missing legs are exactly the T-100 survivors:
`runs_verify/2026-08-18_1328/papers/PMC12096016/research/final_mapped.json`,
`.../PMC12096016/strict/final_mapped.json`, `.../PMC12452463/strict/final_mapped.json`.

**Both reproduce identically at base and at tip**, so **neither is attributable to C-050j** — and its merge
does not change them. Both were red before C-050j existed.

### ⚠ The classification correction, and it matters

C-050j's report listed both among *"three additional `.env`-conditional reds"*. **The load-bearing half of
that claim is true and was verified — they reproduce identically at base.** But **the mechanism is wrong:
they are red with or without `.env`.** They have nothing to do with the PathBank DB.

That distinction is not pedantic. An `.env`-conditional red is an artefact of the measurement environment
and is correctly pre-charged and ignored. **These two are the code telling the truth**, and filing them
under the `.env` family is how a real signal gets permanently silenced.

**Only the third of the writer's three is genuinely `.env`-conditional** —
`tests/test_canonicalization_preflight_and_species.py::test_preflight_warns_when_no_db_and_no_covering_index`,
which asserts on `db_resolution.available`. **It is a seventh member of the `.env` family** and is added to
the standing pre-charge list.

### The tripwire fired correctly. That is the finding.

`test_the_golden_covers_every_committed_leg_fixture`'s own docstring reads:

> *"NEW ACCEPTANCE. A new committed leg must be added to `GOLDEN` deliberately."*

**It is not broken. It did exactly its job** — it detected that the T-100 run committed three legs into the
corpus without adding them to the C-040/C-045 export golden, and it has been red ever since `8ea52c4`.

**The consequence: `build_pwml_ir`'s output is unenforced on three committed legs**, including **both**
strict legs of the two T-100 papers — the same legs every open finding from F-055 to F-064 is reasoned
about. Any card that reads those artifacts is reasoning about output that no golden pins.

### Why no gate caught it — and this is F-054 again

**Both files are in NO chunk.** Certified stem-exactly against `TEST_MATRIX.md`'s chunk tables and
`evidence/chunk_d_gate.py`. Neither is in SMOKE. **So no gate any card is told to run contains either one.**
They surfaced only because C-050j's charter named 14 chunkless files by hand and one reviewer read the
output carefully instead of pattern-matching it to the pre-charge list.

**This is the second-order cost of F-054 made concrete:** 119 of 147 test files sit outside every gate, so a
genuine unconditional red can stay red indefinitely and be mistaken for pre-charged noise.

### Owner and remedy — UNOWNED, and deliberately not fixed here

The remedy is **not** to re-point either assertion. `assert len(_corpus()) == 32` becoming `== 35` is a
one-character edit that would silence the alarm without answering the question it raised. **The question is
whether the three T-100 legs belong in `GOLDEN`**, and answering it means generating their expected
`build_pwml_ir` output and reviewing it — which is precisely the deliberate act the test exists to force.

**REV-051's precedent binds the shape of any fix**: re-pointing an assertion so it passes in both
configurations destroys the property it guards (that is why F-065 must not be "fixed" casually).

**Needs a card.** It must decide, with biological review, whether each of the three legs enters `GOLDEN`,
and it must move the corpus pin **deliberately, with an exact documented delta** under merge rule 4 — never
absorbed. **Registered, not fixed. No accepted card is reopened.**

### Added to the standing pre-charge list — with its mechanism, not just its name

**Unconditional, and NOT `.env`-related** (these are the two above; they are **real** and must not be
silently ignored — cite F-069 when you see them):
`test_c030_canonical_identity_fallback.py::test_the_census_reproduces_over_the_committed_corpus` ·
`test_compound_resolution_extraction.py::test_the_golden_covers_every_committed_leg_fixture`

**`.env`-conditional family, now SEVEN** (four in `test_prefreeze_third_export_seam.py`, one in
`test_prefreeze_species_resolution.py`, **F-065** in `test_pwml_writer.py`, and the newly confirmed
`test_canonicalization_preflight_and_species.py::test_preflight_warns_when_no_db_and_no_covering_index`).

**Line-number drift measured today** (predicate text exact, addresses stale):
`test_prefreeze_species_resolution.py`'s `db_available is False` predicate is at **`:146`**, not `:131`;
`test_prefreeze_third_export_seam.py`'s live `db_resolution` assertions are at **`:378, :383, :414, :436,
:441, :468, :494`**, not `:365/:394/:452`.

## F-070 — the RAG loop reports an OPERATIONAL FAILURE when it merely hit its configured round ceiling, and the default ceiling is 1

- **Severity** **HIGH** · **Registered 2026-08-20**, integration `87dbcca`
- **Surfaced by C-055**, which recorded it as a design note and correctly fed no denominator from it.
  **Independently verified by the orchestrator** against `src/t2pw/rag/loop_policy.py`.
- **⚠ It becomes REACHABLE IN PRODUCTION the moment C-055 merges.** Until then `run_rag_loop` has zero
  production callers and the conflation is inert.

### The mechanism, read from source

`loop_policy.py:139-143` — **one property, two unrelated bounds:**

```python
@property
def out_of_budget(self) -> bool:
    """No further round is affordable. The round bound and the time bound each
    stop the loop independently, which is what makes the loop bounded."""
    return (self.rounds_completed >= self.max_rounds
            or self.deadline - self.now <= self.next_round_reserve_seconds)
```

`loop_policy.py:170` then maps it straight onto D-005's operational-failure reason:

```python
BUDGET_EXHAUSTED: state.out_of_budget and not exhausted,
```

And the module's own docstring states what that reason means (`loop_policy.py:11`, `:18-20`):

> *"`budget_exhausted`, which D-005 counts as an OPERATIONAL failure in pipeline-completion …
> **D-005 names `budget_exhausted` as THE operational-failure denominator.**"*

`DECISIONS.md:134` confirms it: *"`budget_exhausted` is an operational failure in pipeline-completion."*

### Why the two bounds are not the same thing

* **The time bound** — `deadline - now <= next_round_reserve_seconds` — is a genuine resource exhaustion. The
  run wanted more and the clock refused. *Operational failure* is the honest label.
* **The round bound** — `rounds_completed >= max_rounds` — is a **configuration ceiling being honoured**. The
  loop did exactly what it was told. Calling that an operational failure reports a **policy success as a
  malfunction**.

`_conditions`' own comment at `:166-169` states the standard the round bound fails:

> *"`budget_exhausted` asserts D-005's 'another recovery step MIGHT HAVE HELPED': a rung never ran, or the
> round produced claims a further round would integrate."*

When `max_rounds` is reached, another recovery step **might well have helped** — which is exactly why the
reason fires. But the thing that prevented it was **an operator's configured ceiling, not an exhausted
budget**, and D-005's denominator is meant to count runs the pipeline could not complete, not runs it was
told to stop.

### ⚠ Why this is HIGH and not a labelling nicety: the default is 1

`controller.py:237` defaults **`max_rounds=1`**. So on the default configuration, **every RAG loop that
completes its first round and would have continued reports `budget_exhausted`** — an operational failure —
into D-005's denominator. Not an edge case: **the default path.**

The exemption at `:164`/`:170` (`and not exhausted`) only suppresses it when the ladder completed *and*
`new_admissible_claims <= 0`. A round that **did** produce integrable claims — the interesting case, and the
one T-103 exists to exercise — is precisely the one that gets mislabelled.

### Scope, measured — the blast radius is smaller than the name suggests

`budget_exhausted` is a **shared string** across three subsystems, and they are **independent**:

* `pipeline/deadline.py:84` defines its own `BUDGET_EXHAUSTED` and
  `OPERATIONAL_TERMINATION_REASONS = frozenset({BUDGET_EXHAUSTED, OPERATION_TIMEOUT})` (`:120`) — this is
  the **pipeline** deadline seam and is **not** fed by `loop_policy`.
* `batch/driver.py:1701` and `batch/runner.py:921` reference it in **docstrings only**.

**No live consumer converts `loop_policy`'s reason into a rate today** — consistent with C-055's report that
it feeds no denominator. **The defect is that the reason is wrong at the source**, and the moment anything
downstream starts counting termination reasons per leg — which is what T-103 and T-104 will make tempting —
it counts configured stops as failures.

### Owner and remedy — UNOWNED, and deliberately not fixed here

`rag/loop_policy.py` is **C-043's, merged and reviewed**, and it is **outside C-055's boundary** — C-055 was
right to register rather than fix. The remedy is a narrow card that **splits the round bound from the time
bound**, giving the configured-ceiling stop its own reason distinct from D-005's operational-failure set.

**Do not fix this by widening the `and not exhausted` exemption** — that would suppress the reason on runs
where the time bound genuinely fired, trading a false positive for a false negative in the denominator that
matters more.

**Binding on T-103:** its acceptance (`TEST_MATRIX.md:480`) is *"every RAG round re-entered normalization,
mapping, gates, persistence, classification"* — structural, and **unaffected**. But **no operational-failure
rate may be quoted from a T-103 or T-104 run until this is fixed**, because at `max_rounds=1` the reason
fires on the default path. Registered, not fixed. No accepted card is reopened.

## F-071 — an agent's own wall-clock kill leaves a G11 reservation that turns whole-tree G11 red; four occurrences, no durable fix

- **Severity** MEDIUM (process, recurring) · **Registered 2026-08-20**, integration `87dbcca`
- **Fourth occurrence.** Every one was resolved by hand. There is no mechanism that prevents a fifth.

### The mechanism

`g11_evidence.py next --task X --label Y` **creates a real placeholder file on disk** before the job runs:

```json
{ "g11_reserved": true, "task": "X", "label": "Y" }
```

That is deliberate and correct — it is what stops an agent hand-writing a `--json` path and abandoning the
audit trail. The README and the allocator both say so.

**The failure mode it creates:** `bounded_run.py` writes the completed report at the *end* of a job. If the
**agent's own tool wall clock** kills the parent shell first, the wrapper never writes, and the reservation
survives with three keys and **none** of the ~28 fields a completed report carries. `check` then reports it
non-compliant, **whole-tree G11 exits 1, and merge gate 10 fails** — on a job that never produced a result
either way.

**Note the asymmetry that makes this hard to see coming:** `bounded_run.py`'s own `--timeout` is honoured
correctly and its Job Object (`KILL_ON_JOB_CLOSE`) tears descendants down cleanly — **every one of the four
kills left `FINAL SURVIVING COUNT: 0`.** Nothing leaks. The only residue is the bookkeeping artifact.

### The four occurrences

| # | Reservation | Killed by | Disposal |
|---|---|---|---|
| 1 | `evidence/g11/C-042r/09-r1focused.json` | — | PACK 3 **RULING 1**, quarantined to `sprint-records/p3-01-…` |
| 2 | `evidence/g11/REV-050j/10-sample-census-tip.json` | reviewer's 2-min shell limit under CPU/DB contention | PACK 9, quarantined to `sprint-records/p9-01-…` |
| 3 | `evidence/g11/C-055/05-focused-full-r1.json` | agent's 2-min wall clock | `git rm` in C-055 `5ad0d47` |
| 4 | `evidence/g11/C-055/10-c055-own-tests.json` | agent's 6m40s wall clock | `git rm` in C-055 `5ad0d47` |

**#3 and #4 differ from #1 and #2 in a way that matters: they were COMMITTED**, not untracked. PACK 3's
quarantine procedure is written for an untracked file (*"it is in no commit, so moving it cannot alter
history"*) and **does not transfer** to a committed one. The disposal there is a normal `git rm` on the card
branch by the card's own author, with the reason stated in the commit message.

**Occurrence #4 was never disclosed by the agent** — it was found only because the orchestrator's correction
instruction told the author to run `check --task C-055` and report the result. **The author then disclosed it
unprompted and disposed of both in one commit.** That is the process working, but it worked because someone
asked for the check, not because anything enforced it.

### Why this is worth a card rather than a habit

Every remedy so far has been a human noticing. The failure is silent at the moment it happens — the agent
sees its shell die, not a corrupted audit trail — and it surfaces one step later, as a **merge gate failure
attributed to whoever runs the gate next.** Twice it was found by the orchestrator immediately before a
merge.

**Candidate remedies, none owned:**

* a `g11_evidence.py release --task X --label Y` subcommand, so an agent whose job died can dispose of its own
  reservation without a `git rm` and without an orchestrator ruling;
* `check` tolerating a reservation younger than N minutes, so a live job in flight is not reported
  non-compliant — this one needs care, because it would also hide a genuinely abandoned reservation for N
  minutes;
* `next` writing to a staging path that `bounded_run.py` promotes on completion, so an unfinished job leaves
  nothing in the reports tree at all. **This is the only candidate that removes the failure mode rather than
  papering over it.**

### Binding on every agent until it is fixed

1. **Allocate the `--json` path only when the job is ready to start.** Already in the shared execution block;
   it is necessary and demonstrably not sufficient.
2. **Keep `bounded_run.py --timeout` comfortably under your own tool's wall clock**, so the wrapper ends the
   job and writes its report rather than your shell being killed mid-flight. **All four occurrences are the
   opposite ordering.**
3. **Run `g11_evidence.py check --task <YOUR-CARD>` before you report**, and disclose the result. A card that
   reports a clean tip while leaving a non-compliant reservation is handing its reviewer a red gate.

Registered, **unowned, not fixed.** No accepted card is reopened, and no completed report has ever been moved,
deleted or altered in any of the four disposals.

## F-072 — a failed mutex acquire does not stop the job, and the same shell short-circuit has now cleared a live holder's lock

- **Severity** **HIGH** (process) · **Registered 2026-08-20**, integration `c3fd041`
- **Disclosed unprompted by REV-058**, which stopped immediately and reported rather than continuing.
- **Second occurrence of this exact mechanism.** The first was a near-miss the previous session recorded;
  this one actually cleared a lock.

### What happened, in the reviewer's own account

It ran, as one command:

```
mkdir C:/t/heavylock && echo ACQUIRED
<probe>
rm -rf C:/t/heavylock
```

`mkdir` **failed** with `File exists` — C-056d held the lock. **The `&&` suppressed only the `echo`.** The
following statements were separate, so they ran anyway. Consequently:

1. a ~3-second read-only probe ran **while another card held the mutex** — a concurrency violation, minimal
   contention harm;
2. the trailing `rm -rf` **deleted a lock the agent had never acquired**, and because it never read the
   holder file, it could neither name nor restore the holder.

### Why the guard that exists did not help

`_SHARED_EXECUTION_BLOCK.md` § 1 already says *"Clearing another holder's stale lock is the ORCHESTRATOR'S
DECISION ALONE"* — and the agent was not trying to clear anyone's lock. **It believed it held the lock it was
releasing.** The rule is about intent; the failure was about control flow. **A rule against deliberate
clearing does not protect against an unconditional release after a failed acquire.**

`mkdir` on an existing directory is the *correct* mutex primitive — it is atomic and it fails when held. The
defect is entirely in how its failure is consumed.

### Blast radius, measured

**Nil for every measurement.** The orchestrator verified at the time that **no other heavy job was running**,
so nothing contended with the probe, and C-056d's in-flight focused run had nothing to be perturbed by.
**A cleared lock invalidates the protocol, not the result** — C-056d was told explicitly not to discard or
re-run its job.

REV-058's own 12 evidence jobs each acquired and released correctly; only the 12th was affected, and it was a
read-only probe.

### The two rules that actually prevent it — binding on every agent

1. **Never chain an acquire to anything with `&&`, and never let a job follow a failed acquire.** Make the
   acquire its own statement, **test its result explicitly**, and only then run. Compound lines are how
   execution gets past a failure that should have stopped it. `_SHARED_EXECUTION_BLOCK.md` § 10 already
   forbids compound `git add && commit` for a related reason.
2. **Never `rm -rf C:/t/heavylock` unconditionally.** **`cat` the holder file first and remove the lock only
   if it names you.** If it names someone else — stop and report. This is the rule that converts the failure
   from silent to visible.

### Owner and remedy — UNOWNED

The durable fix is to stop making every agent hand-roll a two-phase protocol in shell. Candidates:

* a tiny `heavylock.py acquire --holder <CARD>` / `release --holder <CARD>` pair that **refuses to release a
  lock whose holder file names someone else** and exits non-zero on a failed acquire, so the shell's own
  error handling stops the job;
* or folding acquisition into `bounded_run.py`, which already owns job lifecycle, timeouts and cleanup —
  it is the natural place, and it would make "one heavy job at a time" an enforced invariant rather than a
  convention every agent re-implements.

**The second is the stronger candidate** — it removes the primitive from agent hands entirely, exactly as
`bounded_run` already removed process cleanup from them.

**Related, and the reason this keeps costing:** this is the **third** distinct process defect this sprint
rooted in agents hand-rolling infrastructure in shell — F-071 (reservation left by a wall-clock kill),
PACK 9 RULING 5 (reviewers cannot create the lock at all, so they run unprotected), and now this.
**The protocol asks every agent to re-implement the same three-step dance, and agents get it wrong in a new
way each time.**

Registered, not fixed. **No measurement is invalidated and no card is reopened.**

## F-073 — `runner.CHILD_IMPORTS` is missing six deferred imports, and closing the gap turns a pre-charged red green

- **Severity** MEDIUM · **Registered 2026-08-20**, integration `416e138`
- Surfaced by **C-056d**, which **reported rather than fixed** — the correct call, and the reason this has an
  owner instead of being silently absorbed.

### The mechanism

`tests/test_batch_preflight.py::test_every_import_driver_defers_is_covered_by_the_preflight` asserts that
every import `batch/driver.py` defers into a function body is declared in `runner.CHILD_IMPORTS`, so the
batch preflight can validate them in the child process before a leg starts.

It is **pre-charged red**: its `missed` list already held **4** entries before C-056d existed. The card's new
private helper defers one `release_status` import and one `strict_quarantine` import — **matching the house
style of three neighbouring functions in the same file** — taking `missed` to **6**.

**The test outcome does not change: red at base, red at tip.** What changes is the size of the list it
reports.

### Why the card was right not to fix it

The cure is to add the entries to `runner.CHILD_IMPORTS`. **`runner.py` is C-032's and outside C-056d's
boundary.** More importantly: adding all six would take the test from **red to green**, and *"a pinned
baseline moved deliberately, with an exact documented delta"* (merge rule 4) cuts both ways — **a card
turning someone else's pre-charged red green is absorbing a baseline move it does not own**, and it would
erase the record of why those four were missing in the first place.

**Ratified by the orchestrator as a documented baseline move: `missed` 4 → 6, both additions being the
card's own deferred imports, no other assertion in the file touched.**

### What the finding actually is — and it is not the two new entries

**Four imports were already unregistered before any of this**, which means the batch preflight has been
unable to validate them in the child process for as long as they have existed. That is the defect. C-056d
made it two larger and visible; it did not create it.

**Needs a card that owns `runner.py`.** It must add all six, take the test red → green **as an explicit,
documented baseline move**, and record what the four pre-existing entries were and why they were missed —
because that history is the only evidence of how long the preflight has been partially blind.

**Do not let a future card absorb this quietly in passing.** Registered, not fixed. No accepted card is
reopened.

## F-074 — a replica instrument cannot witness production behaviour, and this is the third time it has produced a false record

- **Severity** **HIGH** (method) · **Registered 2026-08-20**, integration `9492744`
- Found by **REV-059** while rejecting C-059. **The card's code was correct; its evidence artifact was not.**

### The concrete instance

`evidence/c059_leg_replay.json` commits `"dedup_alone_is_payload_neutral": true, "dedup_alone_payload_delta": []`.

Measured through the **real `synthesize_with_report`**, the payload is **not** neutral: the delivered row's
`provenance_lineage` goes **2 entries → 1**. The per-gap `fills_named_gap_directly: via <metabolite>` record
for the losing gap disappears from the payload, and it is **not recoverable from the admission report
either**, because `_reject` (`admission.py:3765`) overwrites `candidate.reasons` with the duplicate code.

**Why the card's instrument could not see it.** Both `tests/test_rag_already_covered_gap.py:767-813`
(`_rows_through_the_production_carry`) and `evidence/c059_already_covered_probe.py` construct `_Reaction`s and
call `_reaction_row` directly — **without ever executing the production line
`reaction.lineage.append(entry.as_dict())` at `synthesize.py:1971-1973`**. `_reaction_row` (`:1652-1663`)
then emits only the `rag_retrieval` entry, which is derived from `gap_ids` and **is** identical once the
union carry lands.

**The replica reproduced every field that did not move and none of the field that did.** It was not a weak
test; it was a test of a different object.

### Why this is HIGH and not a one-off

**Third instance this sprint of the same method failure**, each in a different subsystem:

* **C-050i's census** replayed committed `final_mapped.json` — state from *before* `_apply_create_defaults`
  runs — so it **structurally could not observe** the post-freeze collision C-050j was later chartered to
  guard. Recorded in D-050 §2.
* **`bounded_run.py` retains exit codes but not stdout**, so no pytest summary is recoverable from a G11
  report afterwards. The remedy — capture verbatim summary lines into a committed artifact — is already in
  the shared execution block § 3 as "the instrument gap".
* **This one.**

In each case a **cheaper stand-in for the production path** was measured, the stand-in agreed with the real
thing on everything the author thought to check, and the disagreement lived exactly where nobody looked.

### The rule this makes explicit

**An evidence artifact that asserts something about production output must be produced by running the
production entry point.** A helper that assembles the same dataclasses and calls the same leaf function is
**not** that entry point — it omits whatever the caller does between them, and that is precisely where
regressions hide.

Where a replica is genuinely necessary (cost, determinism, an input that was never written), it is
**acceptable only with a stated scope limit on the artifact itself**: name the production path it stands in
for and the fields it therefore cannot witness. **An unqualified `delta: []` from a replica is a claim the
instrument was never able to make.**

**Corollary for reviewers, which is how this was caught:** when a card commits an artifact asserting
*"identical"*, *"neutral"*, or *"empty delta"*, **re-derive it through the real function before accepting
it**. REV-059 did exactly that and found the residual in one run.

### Owner and remedy

**C-059 is correcting its own artifact under correction round 1** — restating the measured residual, adding
one arm through the real `synthesize_with_report`, and refreshing a stale `g9_proof` count captured at an
earlier tip. **That closes the instance, not the class.**

The class remedy is a line in `_SHARED_EXECUTION_BLOCK.md` § 3, beside the existing stdout gap: **evidence
about production output comes from the production entry point, or carries an explicit scope limit.**
Registered, unowned, not fixed — the shared block is not a card's to edit.

## F-075 — C-059 removes the trigger of the only structural refusal on the F-062 leg, so F-062's own proof fixture may no longer demonstrate it

- **Severity** **HIGH** (sequencing) · **Registered 2026-08-20**, integration `a0bcc0c`
- Measured by the orchestrator directly from the committed T-100 artifacts. **Registered BEFORE the F-062
  charter is written, because it changes what that card can use as evidence.**

### The measurement

`runs_verify/2026-08-18_1328/papers/PMC12096016/strict/quarantine_report.json`:

```
refusal_reasons     : ['degree_zero_export:1']
degree_zero_exports : [{'bucket': 'proteins', 'name': 'Isochorismatase (EntB)'}]
release.status      : diagnostic_only
minimum_core_satisfied : True
```

**`degree_zero_export:1` is the leg's ONLY structural reason, and the single row producing it is
`Isochorismatase (EntB)`** — the duplicate protein created by the RAG re-import that **C-059 now refuses**
(`REASON_ALREADY_COVERED`, which takes `counts.accepted` on that leg from **2 to 0**).

### The consequence, and why it is not good news for F-062

**F-062's mechanism remains real and remains wrong.** `strict_quarantine.py:2013-2034` builds
`structural_reasons` and appends them to `refusal_reasons` **unconditionally** — `review_reasons` receives
only `coverage_reasons`, and only when `defensible_core` holds. So `ok = false`, and
`classify_release_status` pins the leg at `diagnostic_only` before any coverage branch can be reached.
**That is a code defect independent of any leg**, and this finding does not soften it.

**But the leg that demonstrated it may stop demonstrating it.** With `Isochorismatase (EntB)` never
imported, this leg should produce **no `degree_zero_export` at all**, hence no structural reason, hence
`ok = true` — and it would then classify through the coverage branch on its own, with
`minimum_core_satisfied: True`.

**Three things follow, and an F-062 card must not assume any of them away:**

1. **F-062 cannot use this leg as its base-failing G9 proof without re-measuring.** The committed artifact
   was produced *before* C-058 and C-059. A card that replays it and finds `diagnostic_only` will be
   measuring pre-C-059 state; a card that re-runs the leg may find the refusal simply gone. **Neither is a
   proof of the refusal-seam fix.** It will likely need a **synthetic** fixture that produces a structural
   reason on a leg with a defensible core — which is exactly the shape the seam mishandles.
2. **The T-100 acceptance failure attributed to F-062 may resolve without touching the refusal seam.**
   `TEST_MATRIX.md:477` requires PMC12452463 → `review_required`; D-055 recorded both strict legs landing on
   `diagnostic_only` *because of* this seam. If C-059 removes the structural reason, the classification may
   change on its own. **Do not quote that as F-062 being fixed — it would be the trigger removed, not the
   seam corrected.** The next leg with a genuine structural reason and a defensible core hits it again.
3. **This does not unblock T-104, and it does not weaken the merge-rule-6 lock.** The lock existed so the
   fabricated `EntE` and the LDH-derived NAD+/NADH would not ship on a newly-exportable leg. **C-058 and
   C-059 have now landed, which is what discharges it** — not this finding.

### What the F-062 card must do first

**Re-measure the leg on merged integration before writing a single line.** Specifically: does
`degree_zero_export` still appear once C-058 and C-059 are in? If it does not, say so, and build the proof
on a fixture that isolates *the seam* rather than *this leg's biology*.

**And the standing product-owner block still applies:** `PRODUCT_CONTRACT.md:341` conditions PMC12452463's
required outcome on *"after the index fix"*, a phrase occurring **exactly once in the whole control plane
with no antecedent**. Until it is named or struck, **no card can determine whether that locked row binds**,
and F-062 cannot be quoted as contradicting a locked position. **F-062 is therefore blocked on a product
decision, not only on C-058 and C-059.**

Registered, not fixed. No accepted card is reopened.

## F-076 — the c011 golden's regeneration block will silently absorb C-057's delta, because that delta is one-way

- **Severity** **HIGH** (latent) · **Registered 2026-08-20**, integration `b011588`
- **Found and flagged by C-057 itself**, which correctly declined to fix it — `__main__` is outside the
  boundary I granted, and I did not extend it.

### The asymmetry

`tests/test_c011_freeze_seam_golden_equivalence.py` carries per-card helpers that state a baseline move
instead of absorbing it into the fixture — C-030's `_with_c030_hash_keys` (`:236-258`), C-052's
`_with_c052_path_keys` (`:281-302`), and now C-057's `_with_c057_lineage_hashes`.

**Two of the three are reversible. The third is not, and the difference is structural:**

* **C-052's keys are ADDITIVE**, so `_without_c052_path_keys` can strip them and recover the original
  document exactly.
* **C-057's helper REPLACES a digest** — `canonical_payload_sha256`, and the `payload_sha256` inside
  `final_stage3_gate_report` that records the same value a second time. **The pre-move digest is stored
  nowhere else in the fixture**, so there is nothing to restore it from. **No `_without_c057_*` is
  expressible.** This is not an omission by the card; it is a property of a replaced value.

### The consequence

**Running that file's `__main__` regeneration block absorbs C-057's delta into the tracked fixture** —
turning a *stated* baseline move into a silent one, which is exactly the failure the helper mechanism exists
to prevent, and the same class REV-051 refused when it would not let F-065's assertion be re-pointed.

**The golden is named as a BEFORE document.** Once regenerated with the lineage hashes folded in, it stops
being one, and **every earlier card's stated delta becomes unverifiable at the same stroke** — because the
document those deltas are stated *against* no longer exists.

### Why it is HIGH despite nothing being broken today

Nothing is wrong in the tree right now. The hazard is that the trigger is a **routine, sanctioned action**:
someone regenerating the fixture for an unrelated legitimate reason destroys three cards' evidence as a side
effect, and the suite goes green, so **nothing reports it.** A latent trap whose trigger looks like
maintenance is worse than a red test.

**In-boundary mitigation, already taken by C-057:** the limitation is written into the constant's comment,
where a reader meets it *before* running the block — stating that the delta is one-way, that no
`_without_c057_*` is expressible and why, and that `__main__` will silently absorb it.

### Owner and remedy — UNOWNED

Needs a card owning `tests/test_c011_freeze_seam_golden_equivalence.py`. Candidate remedies:

* **make `__main__` refuse to regenerate while any one-way helper is registered**, requiring an explicit
  override flag — the only remedy that removes the failure mode rather than documenting it;
* or have the helper **record the pre-move digest alongside the post-move one**, making the delta reversible
  and restoring symmetry with C-052's;
* or split the fixture so replaced values live in a separate document from additive ones.

**The first is strongest.** A comment stops a careful reader; it does not stop a regeneration run in a hurry.

**Do not fix this by regenerating the fixture "one last time" and starting clean.** That is the defect,
performed deliberately. Registered, not fixed. No accepted card is reopened.

## F-077 — two quarantine scopes are deliberately unattributed, pinned by tests, and each is blocked on a different decision

- **Severity** LOW (coverage gap, deliberate) · **Registered 2026-08-20**, integration `269cdf5`
- **Declared by C-057 in its own report, and pinned by its own tests**, so a later card changes them on
  purpose rather than discovering them by accident. That is the reason this is a finding and not a defect.

C-057 attributes **excluded process rows** with `stage="quarantine"`. Two scopes that a reader might expect
to be covered are **not**, each for a stated reason:

### 1. The three closure prunes — blocked on a report-schema decision, not on effort

`_prune_entities` (`:1294`), `_prune_locations` (`:1349`) and `_prune_biological_states` (`:1419`) discard the
row object itself (`:1345` and siblings). The only surviving carrier is a **fixed-shape record inside
`removed_entity_report.json`** — so attributing them means **growing that record, which is a report-schema
change, not a lineage write.** The module's own house rule (`strict_quarantine.py:2215-2239`) bumps
`schema_version` for exactly that.

**C-057 did not take it, and `schema_version` stays at 6.** A card that wants these attributed must own the
schema bump and its pin, and disclose it as a deliberate baseline move — the same shape D-054 §8 ratified for
C-056c.

### 2. `QUARANTINED_DISCONNECTED` — its only write site is outside the boundary

That row was copied into `originals` **while still accepted**, and the only site that could attribute it,
`_revalidate_surviving_processes`, sits outside C-057's five-function boundary. **Correctly not taken.**

### Why this is worth a registered finding rather than a code comment

**A deliberate gap and an oversight look identical six months later.** Both scopes are pinned by tests in
`tests/test_strict_quarantine_lineage.py`, so a card that widens them will see a red test rather than a silent
behaviour change — but the *reason* each was left lives only in C-057's report and here.

**Neither is a contract violation today.** `PRODUCT_CONTRACT.md:85-102` §3 binds *"the final pathway"*, and a
row deleted by a closure prune is not in it — the same argument that made C-057's exclusion-only reading
correct (PACK 9 RULING 10). These are completeness gaps in the *attribution*, not gaps in the contract.

Registered, not fixed, unowned. No accepted card is reopened.

## F-078 — an adenylation reaction emits `AMP` as a co-product, which is chemically impossible, and the row claims the paper said so

- **Severity** MEDIUM · `product_contract_violation` (§2) · **Registered 2026-08-21**, integration `931c065`
- Surfaced by the **F-069 biological adjudication** (read-only lane), verified unregistered by the
  orchestrator: `grep -rn -i "DHB-AMP|adenylat|pyrophosphate|\bAMP\b"` over `FINDINGS.md` and `DECISIONS.md`
  returns **one** hit, `FINDINGS.md:1402`, which is F-058 describing EntE as *"an adenylation enzyme"*.
  **The chemistry defect itself has never been filed.**

### The mechanism

`runs_verify/2026-08-18_1328/papers/PMC12096016/strict/final_mapped.json`, the `EntE` reaction:

```
2,3-dihydroxybenzoic acid + ATP  ->  2,3-dihydroxybenzoyl-AMP + AMP
```

**One ATP cannot yield both an adenylylated product and free AMP** — that requires two adenosines. EntE
transfers the adenylyl group to DHB and releases **pyrophosphate**. The emitted co-product is wrong, and the
correct one is absent.

### Why it is more than a stoichiometry slip

The `AMP` compound row carries `provenance_lineage[0] = (paper_extraction, paper_stated, explicit)` — an
assertion that **the paper stated it**. `AMP` occurs in the source text exactly once, inside the enzyme name
*"2,3-dihydroxybenzoyl-AMP ligase"*. **It was never named as a discrete metabolite**, and the gold set
anticipated exactly this shape at `src/t2pw/bench/gold/pinned_v1.json:1852`:

> *"The EntE adenylation is excluded from the connected floor because its product is never named as a
> discrete metabolite."*

So this is the same failure mode as F-058's fabricated `EntE` transporter and the NAD+/NADH rows on the same
leg: **a species derived from an enzyme name or an EC number, then stamped `explicit` as though the span
carried it.** Three instances now share that mechanism on one leg.

### Scope, and what is NOT claimed

Observed on **one** committed leg. **Not measured against current source** — the artifact is historical output
from the 2026-08-18 T-100 run. Whether the current pipeline still emits it is unmeasured, and a card must
establish that before treating it as live. Registered, not fixed, unowned.

## F-079 — a payload asserting a reaction that does not occur was classified `release_ready` with `semantic_evaluation: passed`

- **Severity** **HIGH** · `product_contract_violation` (§13, `PRODUCT_CONTRACT.md:343`) · **Registered
  2026-08-21**, integration `931c065`
- Surfaced by the **F-069 biological adjudication**. Verified unregistered:
  `grep -rn -i "release_ready.*semantic|semantic_evaluation.*passed" docs/pwml_recovery_sprint/FINDINGS.md`
  returns **nothing**.
- **Distinct from F-055**, which concerns the batch driver discarding the verdict on **gate-failed** legs.
  **This leg failed no gate.**

### The mechanism

`runs_verify/2026-08-18_1328/papers/PMC12096016/research/final_mapped.json` is the structurally healthy leg of
the three F-069 legs — `quarantine_report.json -> ok: true`, `refusal_reasons: []`, `strict_invariants.ok:
true`, fully accessioned (EntC `P0AEJ2`, EntB `P0ADI4`, EntA `P15047`, EntE `P10378`, EntF `P11454`,
EntD `P19925`). Its `release` block reads:

```
status: "release_ready"          strict_gates_passed: true
semantic_evaluation: "passed"    strict_acceptance_eligible: true
```

`review_flags.json` contains **exactly one** flag: `Fur`'s missing accession.

**The payload nonetheless contains, unflagged:**

1. **A named enzyme asserted to make a product it does not make.** `EntE-catalyzed adenylation of 2,3-DHB`
   has `inputs: ["2,3-dihydroxybenzoic acid (2,3-DHB)", "Adenosine triphosphate"]` and
   `outputs: ["enterobactin"]`. Adenylation produces DHB-AMP. The reaction's own name says *"adenylation"*
   and its own evidence says *"in order to facilitate its covalent attachment"* — and the row then
   short-circuits the entire NRPS assembly into one step.
2. **A fabricated transporter** — `enterobactin secretion` carries
   `transporters: [{entity: "EntE", provenance: "inferred"}]` on a span reading *"secreted to the
   extracellular environment by a **TolC**-dependent process"*. This instance is **F-058**; its being
   unflagged while the leg is `release_ready` is **this** finding.
3. **A degenerate self-referential evidence string** — the `EntF` assembly reaction lists `EntE` as a second
   catalyst whose entire `evidence` is the reaction's own name concatenated with a truncated fragment.

### Why HIGH

`PRODUCT_CONTRACT.md:343` makes **structured status authoritative** and reserves `release_ready` for
shippable output. **A payload asserting a reaction that does not occur was classified shippable, and the
semantic evaluator passed it.** Merge rule 6 exists to stop gates being weakened to increase PWML production;
this is the same wound from the other side — **a gate that was never sensitive to the defect in the first
place.**

### The one qualification, and it is why nothing shipped

`SUMMARY.txt` for that leg records *"PASSED BUT PRODUCED NO DELIVERABLE"* — **the run wrote no PWML.** So the
violation is in the **classification**, not in a released file. That limits the blast radius; it does not
reduce the severity, because the classification is what `PRODUCT_CONTRACT.md:343` makes authoritative.

### ⚠ Interaction with the F-053 prohibition

F-053 prohibits an affirmative reader of `semantic_evaluation == "passed"`. **This finding is evidence that
the prohibition is correct and should stay.** A `passed` verdict has now been observed on a payload carrying
a false product assignment and a fabricated transporter, so any consumer treating `passed` as positive
evidence would be consuming a value that has demonstrably not earned it. **Do not read this finding as a
reason to lift F-053. It is a reason to keep it.**

### Scope, and what is NOT claimed

Observed on **one** committed leg, and it is **historical output**, not current-pipeline evidence. Whether
today's classifier still passes this payload is **unmeasured**. A card must establish that first —
re-measuring under current source is the first obligation, not the fix. Registered, not fixed, unowned.

## F-080 — "after the index fix" HAS an antecedent; three sessions missed it because every search was restricted to `*.md`

- **Severity** MEDIUM (control-plane, **unblocking**) · **Registered 2026-08-21**, integration `6769ea8`
- **This does NOT overturn `DECISIONS.md` D-055 §6.** That entry asks the product owner to *"name the
  referent or strike the condition."* **This finding supplies the evidence for the naming.** The decision
  itself remains the product owner's, and is unmade until they make it.

### The blocked state, and how long it has held

`PRODUCT_CONTRACT.md:341` (§13, LOCKED) conditions PMC12452463's required outcome on *"after the index fix"*:

> *"Correct outcome **after the index fix** is `review_required` with `strict_acceptance_eligible=false`.
> **Never strict success.**"*

Three consecutive sessions recorded that phrase as having **no antecedent anywhere in the control plane**,
and F-062 — the sprint's highest-product-value finding — has been formally blocked on it. The claim is
recorded at `FINDINGS.md:1509-1510`, `FINDINGS.md:2547`, and `DECISIONS.md:3315-3316`.

### The cause of the miss, stated exactly, because it is the reusable lesson

**All three searches were the same search, and it was restricted to `*.md`.** Both `FINDINGS.md:1510` and
`DECISIONS.md:3316` quote it verbatim:

```
grep -rn "index fix" docs/pwml_recovery_sprint/*.md
```

**The antecedent is in a committed `.py` docstring**, which that glob cannot reach.

### The antecedent

`docs/pwml_recovery_sprint/evidence/probe_downstream_gates.py`:

* **`:1`** — *"How far a leg gets AFTER the **stale-index fix** -- the honest limit of **C-010**."*
* **`:5-6`** — *"Fixing **the index defect (C-010)** is not the same as producing PWML."*
* **`:122`** — `print(f"  1. quarantine (index-fixed) : ok={result.ok}")`

Corroborated across the control plane:

* `LEDGER.md:114` — **C-010 = "p01 stale positional index"**, `MERGED` at **`9e06360`**, branch
  `agent/p01-stale-index`, owning `strict_quarantine.py :: _surviving_processes, _degree_zero_exports,
  quarantine_and_close`.
* `MASTER_PLAN.md:215` and `:394` — `C-010 p01-stale-index`.

### Why this probe, specifically, is the right antecedent and not a coincidental phrase

It is **about PMC12452463**, the same paper as the contract row, and it exists to measure exactly the
question the contract row answers:

> *"On `ORIGIN_SHA`, PMC12452463 passes 1 and 2 and then FAILS 3 with `compound_db_resolution_failed` --
> because `build_pwml_ir` performs live PathBank compound resolution AFTER the canonical graph is frozen.
> That is a separate defect (C-040/C-050/C-051/C-052), not a C-010 shortfall."*

and it states its own purpose as:

> *"It is the guard against overclaiming. … **It also gives T-100's acceptance criterion its evidence
> base.**"*

**T-100's acceptance criterion is `PMC12452463 -> review_required, not strict success`** —
`TEST_MATRIX.md:477`. So the probe, the contract row and the milestone acceptance are three statements of one
thing, and the probe is the one that names the event: **C-010, the stale positional index fix.**

### The reading this supports, and its exact strength

**"the index fix" = C-010, merged at `9e06360`.** The condition is therefore **already satisfied**, and
`PRODUCT_CONTRACT.md:341` binds today.

**Stated honestly: this is an objectively grounded reading, not a definition someone wrote down.** No
document says "the index fix means C-010" in those words. The evidence is that C-010 is the *only* fix in the
sprint called an index fix, that the one artifact using the phrase ties it to C-010 **and** to PMC12452463
**and** to T-100's acceptance, and that no competing candidate exists. **A one-line ratification from the
product owner converts it from a strong reading into a settled fact**, and that is what should be sought —
not a fresh investigation.

### What changes if it is ratified

* **F-062 stops being blocked on an undefined condition.** Its mechanism was never in doubt; only whether the
  contract row currently binds. C-067's charter is written against this reading and says so.
* **F-062 may then be quoted as a live contradiction of a locked position**, which `DECISIONS.md` D-055 §6
  currently forbids.
* **T-104's acceptance becomes well-defined** — it requires PMC12452463 to reach the contractually required
  status, which is unquotable while the condition is undefined.

**If it is instead struck**, F-062's mechanism still stands on its own — it is read from code and reproduces
— but the contract half of its severity goes away, and T-104's acceptance row needs rewriting. **Either
answer unblocks work; the absence of an answer is what does not.**

### The reusable lesson, and it has now cost three sessions

**A control-plane search restricted to `*.md` does not search the control plane.** `docs/` contains committed
evidence code whose docstrings carry load-bearing definitions — `probe_downstream_gates.py` is 40 lines of
prose before its first import. **Search `docs/` including `.py`, and search `tests/` too**, before recording
that a term has no antecedent.

## F-081 — `_degree_zero_exports` resolves names WITHOUT synonyms while every other consumer resolves WITH them, so it refuses connected proteins

- **Severity** **HIGH** · `product_contract_violation` · **Registered 2026-08-21**, integration `23614d9`
- **Supersedes F-062's proposed remedy.** F-062's *mechanism* is correctly read and its reading of merge
  rule 7 is right in spirit. **Its proposed fix — routing structural reasons into `review_reasons` — is
  wrong, and this finding records why.** F-062 is not withdrawn; its remedy direction is.
- Produced by an independent read-only biological adjudication of all five structural reasons, commissioned
  because merge rules 6 and 7 pull in opposite directions at that seam.

### The divergence, derived from source and confirmed by execution

**The closure loop already deletes every removable orphan, using a WIDER name test than the detector that
then reports orphans.**

* `strict_quarantine.py:1524` — the pruner keeps a row if `_entity_name_norms([row]) & keep_norms`, and
  `_entity_name_norms` (`process_normalizer.py:626-636`) is **name ∪ synonyms**.
* `strict_quarantine.py:1940-1942` — the detector flags a row if `_normalize(_row_name(row))` is absent from
  `referenced ∪ exempt`. That is **name only**.

`keep_norms` (`:2086`) and `referenced ∪ exempt` (`:1928-1934`) are the same set. Nothing between the closure
fixpoint (`:2124`) and the detector (`:2151`) mutates entities — verified: `_drop_quarantined_processes`
touches only `processes`, `_reconcile_locked_reactions` writes only top-level keys, and
`evaluate_core_coverage` takes a `Mapping` and appends only to locals.

**Therefore, for any run with `converged == True`:**

> a row flagged by `_degree_zero_exports` survived `_prune_entities` at a fixpoint ⟹ its primary name is not
> in `keep_norms` ⟹ its intersection with `keep_norms` came from a **synonym** ⟹ **the row is referenced,
> and the detector cannot see the reference.**

Executed, with its control:

```
referenced norms          : ['a', 'b', 'exa']
pruner norms (name+syn)   : ['enzyme x', 'exa']
pruner keeps row?         : True
detector norm (name only) : enzyme x
_degree_zero_exports      : [{'bucket': 'proteins', 'name': 'Enzyme X'}]
>>> DIVERGENCE: pruner KEEPS a row the detector calls degree-zero

CONTROL genuine orphan: pruner keeps? False | detector flags it | after prune, detector returns []
```

**The detector is the outlier, not the pruner.** `_build_registry` (`:632-634`) states the module's own
resolution rule: *"Synonyms count, exactly as `validate_registry_references` counts them: a reaction that
says 'NAD' resolves against a compound named 'NAD+' that lists 'NAD' as a synonym."* Synonyms are populated
in production by UniProt enrichment (`mapping/enrich_entities.py:1469`, `:1225`). **Admission, pruning and
the registry all count synonyms; only `_degree_zero_exports` does not.**

### What it costs, on the two committed legs

Both `runs_verify/2026-08-18_1328/papers/PMC12096016/strict/` and `.../PMC12452463/strict/`:

```
strict_invariants.degree_zero_exports = [{'bucket': 'proteins', 'name': 'Isochorismatase (EntB)'}]
strict_invariants.closure_converged   = True
coverage.minimum_core_satisfied       = True      coverage.reasons = []
refusal_reasons = ['degree_zero_export:1']        review_reasons = []
release.status  = diagnostic_only                 strict_acceptance_eligible = False
```

**The identical row on two different papers — a systematic name artifact, not leg biology.**

**The gold set settles it.** `bench/gold/pinned_v1.json`, PMC12096016: `mechanistic_relevance: "core"`,
`expected_export: "strict_exportable"`, and `export_rationale` calls the graph *"a **fully connected**
metabolite chain (chorismate to isochorismate to 2,3-diDHB to 2,3-DHB to activated DHB to enterobactin) with
a named enzyme per step"*. **Gold's own word is "fully connected", and the detector calls a protein in it
disconnected.**

### Why F-062's routing remedy is the wrong repair

Routing `degree_zero_export` into `review_reasons` would make `ok = true` and ship
`pathway.review_required.pwml` **carrying a review reason that says "this protein has no connectivity" about
the catalyst of a surviving reaction.** The indicated remedy handed to a human would be to delete a connected
enzyme. Under **merge rule 6** that is weakening a gate to increase production on the strength of a signal
the module's own registry contradicts.

**The correct repair is one layer down**, and it satisfies both rules at once:

> Make `_degree_zero_exports` resolve names the way `_build_registry`, `validate_registry_references` and
> `_prune_entities` already do — through `_entity_name_norms([row])` against `referenced ∪ exempt`. The
> `exempt` construction at `:1929-1934` needs the same treatment.

Then on a converged run it is **empty by construction**; both legs stop being dropped and reach the coverage
branch **on their own merits** — which is exactly what merge rule 7 asks for — **obtained without touching
the routing policy and without relaxing any gate.** The rule *"no protein exported at degree zero"* stays
fully enforced, by the pruner, correctly; the detector goes back to being the residual assertion
`tests/test_strict_quarantine_release_seam.py:268` already calls it.

### ⚠ THE SECOND SEAM — a trap that would ship a PWML on a `diagnostic_only` run

**F-062 frames this as a one-line fix at `:2230-2233`. It is not, and a one-line fix would be actively
harmful.** `classify_release_status` **independently** encodes the same refusal:

* `strict_quarantine.py:2342-2345` computes
  `strict_gates_passed = (not overlaps and not degree_zero and not unaccounted_locks and converged)` and
  `serializable_without_invention = not unexportable` — **separately from `refusal_reasons`**;
* `release_status.py:492-497` checks both **above** the coverage branch, producing `DIAGNOSTIC_ONLY`.

So moving a structural reason into `review_reasons` alone yields **`ok: true` with
`release.status: diagnostic_only`** — an internally contradictory report. And **`ok` is the PWML production
switch**: `app/streamlit_app.py:4717` returns early with no export when `not quarantine_result.ok`. Flipping
`ok` without the classifier would **ship a final PWML on a `diagnostic_only` run**, breaching
`PRODUCT_CONTRACT.md:343` (*"No final PWML for `diagnostic_only`"*).

**Any card touching this area must be told this explicitly.**

### The other four reasons — all `keep_refusing`, and it is not a blanket answer

* **`entity_type_overlap`** — one normalized name in two buckets; every reference binds by `setdefault` to
  whichever bucket sorts first, **deterministically and arbitrarily**, with no record that a choice was made.
  Fails `PRODUCT_CONTRACT.md:189-197`'s third conjunct (*"representable without guessing"*) — the guess is
  literal. `process_normalizer.py:4566` already raises the protein/complex case as a hard error.
* **`unexportable_entity`** — every member of `_entity_representability`'s failure set requires inventing a
  fact to write the row, `placeholder_claims_real_identity` (a **forged accession**) most sharply. Named
  verbatim in the locked text: *"no defensible connected core, **or serialization would require invention**
  → `diagnostic_only`"*.
* **`unaccounted_locked_reactions`** — `locked_reactions_found` is a bare **count**
  (`pipeline/pipeline.py:1064`), retaining no id list, so the artifact can say *how many* locks vanished and
  never **which**. A reviewer is handed *"3 locked reaction(s) are neither active nor quarantined"* and
  cannot find them. **Nothing they could act on.** *(Caveat: the `max()` against a prior count means this
  can also fire from a stale over-count with nothing actually lost. Either way the accounting needs repair,
  not a routing change — and the prerequisite for ever making it reviewable is an **id list**.)*
* **`closure_not_converged`** — all four loop contributors are monotone-decreasing, so a non-converged run
  is a **mid-reduction snapshot, not a fixpoint**. Every other invariant, `coverage`, and the semantic
  verdict are computed on that unfinished graph — **so `defensible_core` itself is unreliable**, and the
  seam's own precondition for routing to review is not established.

### One correction to a committed test's claim

`tests/test_strict_quarantine_release_seam.py:264-267` claims `entity_type_overlap` cannot fire without
emptying the graph. **Not true in general:** a protein `X` exempt as a component of a surviving complex, plus
a compound `X` referenced by a surviving reaction, gives an overlap with both rows surviving and the reaction
resolving cleanly to the compound — reachable **with a defensible core**, and precisely the dangerous
configuration.

### Confidence, and the one thing that would overturn it

**HIGH** on the divergence theorem and on all five rulings — derived from source and executed.

**MEDIUM** on the claim that the production `Isochorismatase (EntB)` row carried a synonym in `keep_norms`.
The theorem forces it, but **it could not be observed directly, because the quarantine input payload is not
persisted** (the standing instrumentation gap at `FINDINGS.md:1580`). This was verified rather than assumed:
`admitted_payload_hash` on the report is `sha256:b22521ec9dfc4088`, while recomputing over the committed
`final_mapped.json` gives `sha256:7e22a4662dbe2f61` and over `merged_payload.json` gives
`sha256:a88b67690be2da81`. **Neither committed file is the payload quarantine judged**, so `synonyms: None`
in `final_mapped.json` is not evidence about the admitted row.

**The one measurement that settles it:** persist the quarantine input payload, or log
`_entity_name_norms([row])` alongside each `degree_zero_exports` entry, and re-run either leg at pre-C-059
code. **Schedule that before the correcting card is written.** If the flagged row's synonym set is disjoint
from `keep_norms`, the theorem is wrong and there is a third divergence not yet found.

**What would change the ruling itself:** a product-owner decision that **synonym-based connectivity is not
real connectivity**. Then the pruner and `_build_registry:632-634` are the defect, the remedy is upstream
duplicate-identity dedupe (F-057, partly delivered by C-059's `REASON_ALREADY_COVERED`), and
`degree_zero_export` stays `keep_refusing` for a different reason. **Either way the routing seam is not the
surface to change** — which is the most confident part of the adjudication.

## F-082 — the G11 credential scanner has no left word boundary, so an ordinary label word ending in "sk" fails merge gate 10

- **Severity** MEDIUM (infrastructure, **live merge-gate hazard**) · **Registered 2026-08-21**, integration `6938cd8`
- **Surfaced by REV-068**, which had two of its **own** evidence artifacts fail the scan, diagnosed the cause
  correctly, and disclosed it rather than quietly renaming the files. **Independently measured by the
  orchestrator before filing.**
- **Assigned to C-063**, which already owns `g11_evidence.py` and had an open correction round. **It does not
  consume that card's second round** — the defect was surfaced after its charter was written.

### The mechanism

`docs/pwml_recovery_sprint/evidence/g11/g11_evidence.py:100`:

```python
("openai_style_key", re.compile(r"sk-[A-Za-z0-9_\-]{16,}")),
```

**There is no left boundary.** REV-068's job labels were `ondisk-nonvacuity-control-green` and
`ondisk-nonvacuity-new-leg-red`; the substring `sk-nonvacuity-control-green` matches, and `check_report`
returns `possible_credential:openai_style_key`.

```
FAIL .../evidence/g11/REV-068/02-ondisk-nonvacuity-control-green.json
     possible_credential:openai_style_key
FAIL .../evidence/g11/REV-068/03-ondisk-nonvacuity-new-leg-red.json
     possible_credential:openai_style_key
```

### Why it is a hazard rather than a curiosity

**Whole-tree G11 is merge gate 10.** Any committed report whose label contains an ordinary English word
ending in `sk`, followed by a hyphen and 16 or more label characters, turns that gate red — and the failure
names a *credential*, which is the most alarming possible way to be wrong. **`disk-`, `task-`, `risk-`,
`mask-`, `desk-`** are all realistic in a job label; `task-` especially so in a sprint whose allocator takes
a `--task` argument.

### The fix, measured rather than assumed

```
string                             current   with \b   expected
ondisk-nonvacuity-control-green    True      False     FALSE POSITIVE (label)
task-reconciliation-evidence       True      False     FALSE POSITIVE (label)
risk-assessment-baseline-run       True      False     FALSE POSITIVE (label)
sk-abcdefghij0123456789ABCD        True      True      TRUE POSITIVE (real key shape)
key=sk-abcdefghij0123456789ABCD    True      True      TRUE POSITIVE (assigned)
```

`\bsk-` removes all three false positives and keeps both true-positive shapes — including the assigned form,
because `=` is a non-word character and therefore supplies the boundary.

**The card was asked to audit the other six patterns for the same class of defect and to change only what is
actually wrong** — `aws_access_key_id` already carries `\b` on both ends and `bearer_token` has
`(?i)\bbearer`, so the defect is not uniform.

**A one-directional regression test would be worse than none here**: a pattern loosened until it matches
nothing would pass a false-positive-only test. The required arm asserts **both** that the label shapes are
clean and that the true-positive shapes are still caught.

### Disposition of the two non-compliant artifacts

`REV-068/02` and `03` are **compliant records of real runs** whose *labels* trip the scanner; the runs
themselves are sound and are **superseded by `08` and `09` under a clean label**, which the reviewer
allocated once it understood the cause.

**They were never committed.** They are left uncommitted rather than deleted, and this record is why — no
completed G11 report has been moved, deleted or altered in this sprint, and that record stands. Once F-082 is
fixed they would pass unchanged, so nothing about them needs correcting.

### Scope note

**This is a scanner defect, not a leak.** No credential appears in any evidence artifact. The scan is doing
its job in the sense that matters — it has never been shown to miss a real key — and this finding narrows a
false positive without weakening it. The measured table above is the evidence for that claim, and any fix
must reproduce it.

## F-083 — the SAME name-resolution divergence one layer up, and this one silently DELETES biology

- **Severity** **HIGH** · `product_contract_violation` · **Registered 2026-08-21**, integration `f2471fd`
- **Found and measured by C-067**, which correctly declared it **out of bounds and did not fix it** —
  `_prune_entities` and the closure loop were not in its charter. That is the second time this sprint a card
  has found a second instance of its own defect and reported rather than absorbed it (PACK 9 RULING 9).
- **Same root cause as F-081, different site, and a strictly worse consequence.**

### The mechanism

`strict_quarantine.py:2081-2085` — the closure loop builds `surviving_complex_norms` using **name only**,
while `_prune_entities:1524` keeps rows on **name ∪ synonyms** (`_entity_name_norms`,
`process_normalizer.py:626-636`). Exactly the F-081 divergence, one layer up.

**Measured by C-067 at both base and tip:** a protein complex referenced **solely under its synonym** is

1. **kept** by `_prune_entities`, because the keep test counts synonyms; but
2. **not recognised as surviving** by `surviving_complex_norms`, because that test does not;

so `_complex_component_norms` returns nothing for it, and **every component protein is then pruned as
`degree_zero_after_quarantine`.**

### Why this is worse than F-081

**F-081 is a false refusal — the run stops and says something wrong.** Loud, and nothing ships.

**F-083 is a silent deletion.** The complex survives; its component proteins are removed from the graph as
orphans; the run continues; and the payload that ships is **missing biology a surviving complex declares it
contains.** Nothing refuses, nothing warns, and `degree_zero_after_quarantine` is a *legitimate* reason code,
so the removal looks correct in `removed_entity_report.json`.

**That is the pruner deleting biology**, which is the failure class merge rule 7 exists to prevent — and it
is invisible in exactly the way F-076 warns about: the suite stays green.

### Scope, and what is NOT claimed

* **Measured on a synthetic fixture**, in C-067's probe, at **both** base and tip — so it is **not**
  introduced by C-067 and is **not** fixed by it. C-067's change is confined to `_degree_zero_exports`.
* **Not measured on any committed leg.** Whether any of the 35 committed legs contains a complex referenced
  only under a synonym is **unmeasured**. That measurement is cheap and should be the first thing the owning
  card does — if the answer is zero, the severity is latent rather than live, and the finding should say so.
* C-067's own `exempt` widening is **asymmetric with the closure loop** as a result. C-067 argues this is
  safe because it can only exempt rows the pruner has already deleted, making it a no-op on a converged run,
  and non-convergence refuses independently. **That argument was accepted for C-067's scope; it is not a
  reason to leave F-083 unfixed.**

### Remedy — UNOWNED

Align `surviving_complex_norms` with `_entity_name_norms`, as C-067 did for `_degree_zero_exports`.

**⚠ The subtlety that makes this more than a copy of C-067's patch, and C-067 flagged it explicitly:** the
set must still **store** the primary-name norm, because `_complex_component_norms` keys on it. **Widening
what is stored would silently stop it matching** — trading one silent deletion for another. The fix is to
widen the *test*, not the *stored value*. A card that copies C-067's diff shape without noticing this will
make the defect worse.

### Related, and it is now a pattern rather than an incident

**Three sites, one rule, and only some of them follow it.** `_build_registry:632-634` states the module's own
resolution rule — *"Synonyms count, exactly as `validate_registry_references` counts them"* — and admission,
`_prune_entities` and the registry all honour it. `_degree_zero_exports` did not (**F-081**, fixed by C-067)
and `surviving_complex_norms` does not (**this finding**).

**The owning card should audit every remaining name comparison in `strict_quarantine.py` against that rule
and report the full census**, rather than fixing the one site named here. Two instances found by accident, in
two different cards, is evidence there are more.

### Orchestrator verification at source, and the code says the consequence out loud

Confirmed independently at integration `f2471fd`, `strict_quarantine.py:2081-2085`:

```python
surviving_complex_norms = {
    _normalize(_row_name(row))                                  # <- STORES the primary-name norm
    for row in _safe_list(_safe_dict(working.get("entities")).get("protein_complexes"))
    if isinstance(row, dict) and _normalize(_row_name(row)) in referenced   # <- TESTS name ONLY
}
keep_norms = referenced | _complex_component_norms(working, surviving_complex_norms)
```

**`_normalize(_row_name(row))` appears twice and the two occurrences are not the same obligation.** The
**membership test** is the defect and must be widened to `_entity_name_norms([row]) & referenced`. The
**stored value** must stay the primary-name norm, because `_complex_component_norms:1484` keys on it
(`if _normalize(_row_name(row)) not in surviving_complex_norms`). C-067's warning is exactly right, and a
card that widens both will break the lookup it is trying to fix.

**The consequence is stated by the code itself.** `_complex_component_norms`'s docstring (`:1472-1477`):

> *"Requirement 3: a component protein is kept even at degree zero. It has no edge of its own **by
> construction** -- the complex carries the edge -- so **pruning it on connectivity would gut every surviving
> complex**."*

**That protection is the thing the synonym gap defeats.** The module knows pruning components guts surviving
complexes, built a mechanism to prevent it, and then gated that mechanism behind a name test that the rest of
the module does not use. This is not a subtle consequence being inferred — it is the documented failure mode
of the exact code path.

**A second construction site exists at `:1929`**, feeding the `exempt` set for `_degree_zero_exports`, with
the same shape. **C-067 fixed that one** (it was inside its boundary) and left `:2081` alone, which is why
the two are now inconsistent — disclosed by the card, accepted for its scope, and the reason this finding
should be taken promptly rather than banked.

---

## F-081 — CONFIDENCE UPGRADE (appended 2026-08-21, integration `f2471fd`)

F-081 was registered **MEDIUM** on the claim that the production `Isochorismatase (EntB)` row carried a
synonym in `keep_norms`, because **the quarantine input payload is not persisted** (`FINDINGS.md:1580`) and
the theorem forced the conclusion without observing it.

**C-067 derived it from the committed artifacts alone**, which was flagged as valuable-and-unscheduled and
delivered. From `runs_verify/2026-08-18_1328/papers/PMC12096016/strict/`:

1. `final_mapped.json` — reactions 1 and 5 name the enzyme entity **`isochorismatase`**, and **no protein row
   is named that** (rows are `EntC, EntB, EntA, EntE, EntF, Fur, Isochorismatase (EntB)`).
2. `quarantine_report.json` — **all 9 processes `core_accepted`, `quarantined_broken_reference: 0`.** So
   `isochorismatase` **resolved**, through `_build_registry`, which counts synonyms. A synonym-bearing row
   therefore exists.
3. `removed_entity_report.json` — the closure removed exactly **one** protein, `Fur`, with
   `closure.converged: true`. So `Isochorismatase (EntB)` **survived the fixpoint prune**, whose keep test is
   name ∪ synonyms.
4. The detector nonetheless flagged it ⟹ `_normalize("Isochorismatase (EntB)")` = `"isochorismatase entb"`
   ∉ `referenced ∪ exempt`.

**⟹ its match with `keep_norms` came from a synonym, and `referenced` contains `"isochorismatase"`.**
F-081's theorem, instantiated on the production leg from its own committed report, naming the string.

Mechanism corroborated at `mapping/enrich_entities.py:1462-1469`: protein synonyms = existing ∪ UniProt
`alternative_names` ∪ `gene_names` ∪ `recommended_name`.

**Two caveats, stated by C-067 rather than extracted from it.** The derivation assumes the admitted payload
shares `final_mapped.json`'s **process** rows — the admission pointers and names match exactly (6 reactions,
2 transports, 1 interaction), so the divergence between the two files is in the **entity** rows, which is
where enrichment writes synonyms. And **the synonym string itself was not read**:
`data/enrichment_cache.json` has **no `P0ADI4` entry**, so its provenance is inferred from code, not
observed.

**F-081's mechanism confidence is raised from MEDIUM to HIGH on this leg.** The remaining gap is narrow and
named. **A separate observation worth its own attention:** the enrichment cache holds no entry for an
accession two committed legs map to, so the cache does not cover the accessions those runs actually used.
