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
| F-031 | `MASTER_PLAN.md` § 9 — the **Canonical paths** table, row `extract.py` (anchor, not a line; ~~`:336`~~ struck by C-112 — that address held § 7's heading even at C-109's base) | Lists `src/t2pw/rag/extract.py` as a decoy; it is **live**, reached via a multi-line tuple import at `streamlit_app.py:454` and used at `:645` in `maybe_run_rag` |
| F-032 | `MASTER_PLAN.md` § 9 — the branch-register row `C-035` (anchor, not a line; ~~`:363`~~ struck by C-112 — that address held TRAP-5 even at C-109's base) | Assigns `rag/admission.py` to C-035 (lineage writes), but a future **C-061** needs behavioural changes to `parse_span_relation` and `validate_evidence_span` in the same file. C-035 merged with both **byte-identical** to keep them free; the §9 row still needs fixing before C-061 is dispatched |
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
| A | 6 | `TEST_MATRIX.md` `## Chunks` table, row `**A**` — ~~`:213`~~ drifted, see F-154 |
| B | 6 | same table, row `**B**` — ~~`:214`~~ |
| C | 8 | same table, row `**C**` — ~~`:215`~~ |
| D-core | 5 | `chunk_d_gate.py` symbol `CORE` — ~~`:63-67`~~ |
| D-s8 | 1 | `chunk_d_gate.py` symbol `S8` — ~~`:69`~~ |
| D-qb | 1 | `:70` |
| E | 1 | `TEST_MATRIX.md` `## Chunks` table, row `**E**` — ~~`:218`~~ struck by C-112: that is the `child_env` row of the bounded-runner table; Chunk E's row measured at `:237` |

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

> **⚠ CLOSED by C-070, merged `09f7156`. Two claims above are REFUTED BY MEASUREMENT, and both are left
> standing rather than deleted because they were honest when written.**
>
> **1. *"SMOKE and all four chunks would have to be re-pinned"* is an over-estimate — for the
> `pythonpath = src` remedy only.** Measured, nothing moved. Chunk C **109 → 109** and Chunk D-core
> **160 → 160** with byte-identical **per-node outcome lists**; and across SMOKE's 20 files + Chunk D's 7 +
> Chunk E's 1, the collected node-ID sets are **identical, 834 = 473 + 187 + 174**. REV-070 re-derived this
> by a tighter route than base-vs-tip — holding the tree constant and toggling only `-o pythonpath=`.
> Orchestrator heavy gates on the branch: SMOKE **473**, Chunk A **134**, Chunk E **174**, Chunk D
> **187/187 `failed=none`**. **Nothing was re-pinned, because nothing moved.**
> *Mechanically it cannot move:* `src` is already on `sys.path` in every pinned run from two independent
> sources — the measured launcher **requires** `PYTHONPATH=<tree>/src`, and 132 files insert it themselves.
> The ini adds a duplicate of an entry that is always present, and duplicate `sys.path` entries are inert.
> **The claim is refuted only for this remedy.** A repo-root `conftest.py` and an editable install both
> change `sys.path` construction and import mode in ways nobody has measured; C-070 claimed nothing about
> them and neither does this note.
>
> **2. The 21-file list above is wrong in BOTH directions, and the count is right by coincidence.**
> Measured at `e616846`: **155** test files, **132** mention `sys.path`, **23** do not. Of those 23,
> **18 fail and 5 pass**; of the 132, **3 fail**. 18 + 3 = 21.
> *Named above but collecting alone perfectly well:* `test_pathbank_unknown_fallback`,
> `test_prefreeze_species_resolution`, `test_protein_export_policy`, `test_streamlit_stage8_export_contract`
> — all four reach `src` through `from helpers_prefreeze import ...`. (`test_dependency_declarations` is the
> fifth passer and imports no `t2pw` at all.)
> *Missing from the list and genuinely failing:* `test_c064_round_cap_reason`,
> `test_compound_db_match_admission` (names `sys.path` only in a **docstring**), `test_curator_offline_mode`
> and `test_semantic_production_no_gold` (both use it **inside a test body**, not at import time).
>
> **So no static predicate separates the two sets** — which is why C-070's acceptance test is a real sweep
> over every `tests/test_*.py` rather than a name list, and why its own first design, an "at-risk subset"
> sweep, was discarded: it would have missed 3 of the 21.
>
> **When citing this finding's exposure, use the measured census** at
> `evidence/c070_isolated_collect_base.json`, **not the name list above.**
>
> **One caveat on the census, from REV-070:** an independent base sweep reported **22**, not 21. The extra is
> `tests/test_c030_canonical_identity_fallback.py:88`, which shells out to `git ls-files` at **import** time
> and therefore cannot be collected in a `.git`-less **exported** tree — see **F-089**. The 21 are a strict
> subset and all reproduce.

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

> **⚠ SUPERSEDED 2026-08-21, integration `e616846`. Measured — and the finding SPLITS INTO TWO HALVES with
> different statuses. The paragraph above is left standing because it was true when written.**
>
> **Half A — the chemistry. NOT A CODE PREDICATE, and no card can own it today.** The row
> (`out: ['2,3-dihydroxybenzoyl-AMP', 'AMP']`) is Stage-1 **LLM extraction output**. No deterministic
> function in `src/` produced it, so there is no `file:line` predicate to cite and no offline proof that
> today's pipeline still emits it. Owning this half requires an **authorised LLM leg**, which is a separate
> authorization, not a card.
> *The chemistry claim itself is CONFIRMED* against the committed paper text
> (`runs_verify/2026-08-18_1328/papers/PMC12096016/01_source_text.txt`, 43,667 chars): `AMP` occurs
> **exactly once**, inside the enzyme name *"EntE (2,3-dihydroxybenzoyl-AMP ligase; EC 6.2.1.71)"*, and
> `pyrophosphate` occurs once as **thiamine** pyrophosphate in an assay buffer — unrelated. `PPi`: zero.
> So *"the correct co-product is absent"* is verified, with the added precision that the single
> `pyrophosphate` hit is not the EntE product. **Note `00_PAPER.txt` in that directory is a 709-char stub;
> `01_source_text.txt` is the real text.**
>
> **Half B — the provenance claim. STILL REPRODUCES, deterministically and offline.** Feeding the historical
> `AMP` row through today's `_paper_entry` returns `stage=paper_extraction`, `origin=paper_stated`,
> `paper_explicit=explicit`, `review_required=False` — byte-identical to the committed
> `provenance_lineage[0]`. The predicate is `stage_one_boundary.py:311-315` with
> `_PROVENANCE_NOT_READ = ("inferred", "enriched")` (`:212`) and
> `_MARKS_NOT_READ = ("inference", "rag_provenance")` (`:217`): **any row the extraction did not self-mark is
> stamped `explicit` with no review flag, and the seam has no paper text with which to check.** Its own
> docstring concedes this (`:168-171`): *"this seam receives a payload, not the paper it was drawn from."*
>
> **Half B is BLOCKED ON A PRODUCT DECISION and is deliberately not chartered.** `settle_stage_one`
> (`:411-418`) takes no source-text parameter and its only production call site is
> `src/t2pw/app/streamlit_app.py:5476` — the file carrying the protected uncommitted product-owner edit. So
> the only in-boundary change is to what the seam **claims**, not to what it **checks**, and choosing
> between two claims with no new evidence is policy, not engineering. The question put to the product owner
> is whether `PRODUCT_CONTRACT.md:85-102` §3's *"whether it was paper-explicit"* must be **verified** or
> merely **recorded**. Ownership, G9 posture and the one pinned test that moves
> (`tests/test_stage_one_boundary_lineage.py:246-256`) are all measured and ready if the answer is
> "verified".
>
> **All fourteen consuming test files for this seam are outside the chunk table.** No chunk and no SMOKE
> covers them; a charter must enumerate them explicitly.

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

> **⚠ SUPERSEDED 2026-08-21, integration `e616846`. The obligation above is DISCHARGED — it was measured,
> and the finding STILL REPRODUCES IN FULL.** Feeding the committed `final_mapped.json` through today's
> `evaluate_production_semantics` → `semantic_verdict` → `classify_release_status`, with the request derived
> as the pipeline does (`pathway_context_from_stage_zero` over the committed `manifest.jsonl`
> `observed_context`), returns `evaluation=passed`, `status='release_ready'`,
> `strict_acceptance_eligible=True`, `semantic_failed_checks=[]` — **byte-identical to the four committed
> fields** in `runs_verify/2026-08-18_1328/papers/PMC12096016/research/quarantine_report.json` → `release`.
> No PACK 10 card touched the predicate: `git log 931c065..HEAD -- src/t2pw/bench/semantic_production.py
> src/t2pw/bench/semantic.py src/t2pw/pipeline/release_status.py` is **empty**.
> **Owned by C-071** (`prompts/C-071.md`), which is chartered against this measurement. The paragraph above
> is left standing because it was true when written; it is corrected, not deleted.

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

## F-084 — the `httpx` / `httpx2` cross-stack timeout hypothesis, INVESTIGATED AND DISPROVED offline

- **Severity** LOW · **NOT a defect** · **Registered and closed in the same entry**, 2026-08-21, integration `e616846`
- **Registered so it is not re-investigated.** C-066 raised the possibility as an aside
  (`prompts/C-066.md:29-45`) and it was carried forward as a candidate. It was never a
  registered finding: `grep -rn "F-084" docs/ src/ tests/` — searched `.md`, `.py` and `.txt`
  alike, not `*.md`-only — returned **zero hits** before this entry.
- **No live external request was made, and none is required.** The type boundary was proven
  offline end to end, through `httpx2.MockTransport`, zero sockets.

### The hypothesis

*"OpenAI 3.x may ignore or reject a `httpx.Timeout` object constructed from a different HTTPX
stack."* **False for this codebase.**

### The type boundary is real — verified

Both stacks are installed: `httpx` 0.28.1 and `httpx2` 2.12.0 (with `httpcore` 1.0.9 and
`httpcore2` 2.12.0), alongside `openai` 3.3.1. They are **different class objects**:

```
httpx.Timeout  = <class 'httpx.Timeout'>   __module__='httpx'
httpx2.Timeout = <class 'openai.Timeout'>  __module__='openai'
same class object?                           = False
isinstance(httpx.Timeout(1), httpx2.Timeout) = False
openai._types.Timeout is httpx2.Timeout      = True
```

Both MROs are `['Timeout', 'object']` — no shared base. `openai` imports `httpx2` in 12
modules (`openai/_types.py:36`) and `httpx` in **zero**. `t2pw` constructs the legacy one:
`src/t2pw/llm/client.py:55-59` `return httpx.Timeout(seconds, connect=CONNECT_TIMEOUT_SECONDS)`,
handed to the SDK at `:87` and `:101`.

### Why it is nonetheless correct — `openai` ships an explicit shim

`.venv/Lib/site-packages/openai/_httpx2.py:65-69`:

```python
def normalize_httpx_timeout(value: float | httpx2.Timeout | None) -> float | httpx2.Timeout | None:
    module = _loaded_legacy_httpx()          # sys.modules.get("httpx")
    if module is not None and isinstance(value, module.Timeout):
        return httpx2.Timeout(**value.as_dict())
    return value
```

invoked on every request build at `openai/_base_client.py:610-621`. Its `sys.modules["httpx"]`
lookup **cannot fail on this path**: an `httpx.Timeout` cannot be constructed without `httpx`
already being in `sys.modules`.

**Measured through the real SDK call path, mocked transport, no network:**

```
BaseClient.timeout stored     = Timeout(connect=5.0, read=300.0, write=300.0, pool=300.0)
request.extensions['timeout'] = {'connect': 5.0, 'read': 300.0, 'write': 300.0, 'pool': 300.0}
mocked chat.completions.create -> succeeded; all floats: True
streaming request              -> {'connect': 5.0, 'read': 300.0, 'write': 300.0, 'pool': 300.0}
with_options(timeout=httpx.Timeout(30, connect=5)) -> read/write/pool 30.0, connect 5.0
```

**Exactly what `client.py:55-59` intends: 300 s overall, connect pinned at 5 s.** Neither
ignored nor rejected. No `TypeError`, no `APITimeoutError`, no silent fallback.

### ⚠ The latent sub-finding — real corruption, currently unreachable

`openai/_base_client.py:955-958` hands the **raw** legacy object to the inner client (the
`:928` normalisation is inside the `if not is_given(timeout)` branch and does not apply):

```python
self._client = http_client or SyncHttpxClientWrapper(base_url=base_url, timeout=cast(Timeout, timeout))
```

`_DefaultHttpxClient.__init__` (`:872-877`) only `setdefault`s, so no conversion happens;
`httpx2/_client.py:203` `self._timeout = Timeout(timeout)` then falls to
`httpx2/_config.py:132-138`'s scalar branch and assigns the whole legacy object to all four
fields:

```
client._client.timeout.as_dict() = {'connect': Timeout(connect=5.0, read=300.0, ...), 'read': <same>, ...}
1.0 + client._client.timeout.connect  ->  TypeError: unsupported operand type(s) for +: 'float' and 'Timeout'
```

**Unreachable on every path this product takes**, for three independently measured reasons:

1. `openai/_base_client.py:621` always passes an explicit `timeout=` to `build_request`, so
   `httpx2/_client.py:357-359`'s `isinstance(timeout, UseClientDefault)` fallback never fires.
2. `httpx2/_client.py:560-563` `_set_timeout` is guarded by `if "timeout" not in request.extensions`
   — already set by (1). Redirects preserve it (`httpx2/_client.py:469` copies `extensions`).
3. `openai` reads the inner client's `.timeout` only at `_base_client.py:928` and `:1553`,
   both guarded `if http_client` — and `grep -rn "build_request|\.send\(|with_options" src/ --include=*.py`
   returns **zero hits**; `src/t2pw/llm/client.py` never passes `http_client=`.

Demonstrated directly: driving the raw inner client (`client._client.request(...)`, a path the
repo never takes) **does** hand the transport the corrupted objects; the SDK path never does.

### Disposition

**NOT a `product_contract_violation`. No card. No code change today.**

**The standing risk, stated so it is not forgotten:** the safeguard is a property of `openai`'s
internals, not of anything this repo controls. **An `openai` upgrade could expose it.** If a
card ever does want it closed, the one-line boundary is `src/t2pw/llm/client.py:55-59` —
return an `httpx2.Timeout` when `httpx2` is importable — but that would also require
revisiting C-066's `requirements.txt` declaration, so it is not free.

---


## F-085 — F-080 names the wrong SHA for C-010: `9e06360` is its BASE, and the LEDGER's Merge SHA column says `72ee20f`

- **Severity** MEDIUM (control-plane, **blocks a correct ratification**) · **Registered 2026-08-21**, integration `e616846`
- **Does NOT overturn F-080's reading.** F-080 concluded that *"the index fix"* at
  `PRODUCT_CONTRACT.md:341` means C-010, and that conclusion survives intact. **Only the
  commit identifier attached to it is wrong.** The measurement below in fact *strengthens*
  F-080's central claim.
- **Caught before ratification, which is the whole point.** `DECISIONS.md` is append-only.
  Had the product owner ratified the wording as circulated, a false fact would have been
  written permanently into a LOCKED entry, and every downstream citation of C-010 would have
  inherited it.

### The error

F-080 records, under "The antecedent":

> *"`LEDGER.md:114` — **C-010 = "p01 stale positional index"**, `MERGED` at **`9e06360`**,
> branch `agent/p01-stale-index` …"*

`LEDGER.md:112` is the header row, and its columns run:

```
| ID | Task | Status | Deps | Base SHA | Branch | Worktree | Ownership boundary |
  Reviewer | Focused | Integration | Merge SHA | Bench delta | Blockers |
```

On the C-010 row, `9e06360` sits in **Base SHA**. The **Merge SHA** cell reads **`72ee20f`**.
**The finding read the wrong column.**

### Verified by execution, four independent ways

```
$ git log -1 --format="%h %s" 72ee20f
72ee20f Merge C-010 (agent/p01-stale-index): degree zero answered against the pre-prune snapshot

$ git log -1 --format="%h %s" 9e06360
9e06360 Merge C-012 (agent/p00b-driver-seam): driver.py _drive terminal-path seam

$ git merge-base --is-ancestor 9e06360 72ee20f && echo ANCESTOR
ANCESTOR
```

and the content check, which is the decisive one:

| commit | source files touched |
|---|---|
| **`72ee20f`** | `src/t2pw/pipeline/strict_quarantine.py`, `tests/test_strict_quarantine.py`, `tests/test_strict_quarantine_real_artifact_replay.py`, `docs/change_log.md` — **exactly C-010's declared ownership boundary at `LEDGER.md:114`** |
| `9e06360` | `src/t2pw/batch/driver.py`, `tests/test_batch_driver_seam_golden.py`, and `evidence/g11/**C-012**/*` reports |

`9e06360` writes C-012's own G11 evidence directory. It is C-012's merge, and it is C-010's
base because C-010 was cut from the integration tip that C-012's merge produced. **The two
facts are consistent; the finding simply took one for the other.**

### Why this strengthens F-080 rather than weakening it

F-080's load-bearing claim is that **no competing candidate exists** for *"the index fix"*.
That was argued from the prose. It can now be **measured**:

```
$ git log --oneline --all --merges | grep -i index
72ee20f Merge C-010 (agent/p01-stale-index): degree zero answered against the pre-prune snapshot
```

**One commit in the entire repository, across all branches, whose merge subject contains
"index" — and it is C-010.** F-080's reading is better supported after this correction than
before it.

### The corrected ratification wording

> *"After the index fix" in `PRODUCT_CONTRACT.md:341` refers to C-010, merged as `72ee20f`.*

### The reusable lesson, and it is the sibling of PACK 10 RULING 1

PACK 10 RULING 1 was *"a control-plane grep restricted to `*.md` does not search the control
plane."* This is its sibling:

**A citation lifted from a 14-column Markdown table must name the column it came from.** The
LEDGER's card rows carry **two** SHA columns — `Base SHA` and `Merge SHA` — five columns
apart, on rows thousands of characters wide. Quoting *"`MERGED` at `<sha>`"* without naming
the column is how a base becomes a merge.

**Cheap standing guard, and it would have caught this in one command:** before citing any
sprint SHA, run `git log -1 --format="%s" <sha>` and confirm the subject names the card you
think it does. It costs one line and it is decisive.

---
## F-086 — the batch preflight's own detector discards submodule names, hiding a third undeclared module from the guard that exists to find them

- **Severity** MEDIUM · **Registered 2026-08-21**, integration `e616846`
- **Found by the orchestrator** while independently verifying C-069's predicate rather than
  trusting its charter. **Assigned to C-069**, which already owns both files, with its ceiling
  raised 400 → 650. **It does not consume that card's correction rounds** — the defect was
  surfaced after its charter was written.
- **This is F-073 one layer down.** F-073 is *"the preflight has been partially blind."*
  F-086 is *"and the instrument that measures the blindness is itself blind, in a way that
  reports zero problem."*

### The mechanism

`tests/test_batch_preflight.py :: _deferred_imports` records `inner.module` for an
`ast.ImportFrom` node:

```python
if isinstance(inner, ast.ImportFrom):
    if inner.level == 0 and inner.module:
        found.append(inner.module)
```

For `from t2pw.pipeline import deadline as leg_deadline` (`src/t2pw/batch/driver.py:1718`)
that records **`"t2pw.pipeline"`** — the package — and **discards `deadline` entirely**.
`_covered("t2pw.pipeline")` then returns **True**, because `CHILD_IMPORTS` already lists
`t2pw.pipeline.export_mode`, which `startswith("t2pw.pipeline.")`.

Measured at `e616846`:

```
deferred total (helper view): 9
MISSED: 6            CHILD_IMPORTS entries: 5

line 1718: from t2pw.pipeline import deadline
   -> REAL SUBMODULE t2pw.pipeline.deadline
   ; helper recorded 't2pw.pipeline'
   ; covered('t2pw.pipeline')          = True
   ; covered('t2pw.pipeline.deadline') = False
```

**`t2pw.pipeline.deadline` is deferred in `driver.py`, is absent from `CHILD_IMPORTS`, and the
preflight cannot validate it in the child — while the guard reports zero problem.** A third
distinct blind module, on top of F-073's `t2pw.pipeline.release_status` (4 sites) and
`t2pw.pipeline.strict_quarantine` (2 sites).

### Why it is worse than F-073, and why the direction matters

`_covered`'s own docstring already warns, correctly:

> *"It does NOT cover its own descendants -- importing `t2pw.rag` says nothing about
> `t2pw.rag.research_report`, which is one of the two modules this whole check exists for."*

**The authors knew the descendant direction was unsafe.** The defect is that
`_deferred_imports` **loses the descendant name before `_covered` is ever called**, so
`_covered` is asked about the parent and answers correctly — about the wrong thing.

F-073's failure mode is a guard that finds a real gap and is ignored. **F-086's is a guard
that reports success because it asked the wrong question**, which is strictly worse: it is
indistinguishable from correctness at every level above it.

### The fix direction, and its sharp edge

Resolve a `from <pkg> import <name>` site to `<pkg>.<name>` **when `<name>` is a real
submodule**, and leave it as `<pkg>` when it is an ordinary attribute — a class, a function, a
constant. `importlib.util.find_spec` distinguishes them.

**Getting this wrong in the permissive direction turns every `from x import SomeClass` into a
bogus `x.SomeClass` entry** and floods `missed` with names that are not modules at all. The
card must test both directions.

### The reusable lesson

**A guard that derives its own input is only as good as the derivation, and the derivation is
usually untested.** `_deferred_imports` had no test of its own; the only thing exercising it
was the assertion that consumes its output, which cannot distinguish *"nothing is missing"*
from *"I did not look."* **RULING 13's non-vacuity requirement should extend to a guard's
input derivation, not only to its assertion.**

---

## F-087 — `runner.py:1341`'s stated measurement went stale, and the guard above it survives only because it deliberately asserts by name

- **Severity** LOW (prose / rationale drift, **no behavioural consequence**) · **Registered 2026-08-21**, integration `e616846`
- **Found by C-069** while building its behavioural arm, and **correctly reported rather than
  fixed** — it sits outside that card's `CHILD_IMPORTS`-only boundary. Stale at the base SHA,
  independently of C-069's change.

### The drift

The `#:` block above `CHILD_IMPORTS` (`src/t2pw/batch/runner.py:1338-1361`) states:

> *"Measured 2026-07-28: importing all four of the entries above in a fresh interpreter leaves
> `t2pw.llm.client` absent from `sys.modules`"*

and that measurement is the **stated reason the fifth entry exists**.

**Re-measured 2026-08-21 in a fresh interpreter, per module:** `t2pw.rag.research_report`
**now reaches** `t2pw.llm.client`. (`streamlit.testing.v1`, `t2pw.batch.driver` and
`t2pw.pipeline.export_mode` do not, and neither do C-069's two new entries.) Evidence:
`evidence/c069_probe_coverage_base.json` / `_tip.json`, key `llm_client_reach.per_module`.

### Why nothing breaks — and this is the interesting half

`test_the_llm_backend_is_probed_even_though_nothing_else_reaches_it` asserts **by name**
rather than through `_covered`, **exactly as its own docstring says it deliberately does**.
So the guard survives its own rationale going stale.

**That is the design working, not luck.** A test that had asserted *"nothing else reaches it"*
dynamically would now be green for the wrong reason — the entry would still be needed, but the
test would have stopped proving why. By pinning the name instead, the guard keeps enforcing
the requirement even after the justification for it changes.

What is stale is only the **justification prose**, in two places: `runner.py:1338-1361` and
that test's docstring.

### Disposition

**No card of its own.** The correct remedy is a prose update in both places, re-stating the
measurement with its 2026-08-21 result and noting that the entry is retained because a
transitive reach through one sibling is not a guarantee. **It should ride along with the next
card that owns `runner.py`'s `#:` block or that test file** — not be made a card to make the
queue look busy.

### The reusable lesson

**A comment that cites a measurement should cite its date and its method, so a later reader
can tell staleness from disagreement.** This one did cite its date, which is the only reason
the drift was detectable at all rather than being argued about.

---

## F-088 — the tree-pin guard's own rationale cited `pytest.ini` as setting no `pythonpath`, and C-070 made that false

- **Severity** MEDIUM (control-plane / instrument prose, **no functional effect**) · **Registered and CLOSED in the same entry**, 2026-08-21, integration `09f7156`
- **Surfaced by REV-070**, which noticed that approving C-070 would falsify a sentence in the module that certifies every measured run in this sprint. **C-070 correctly did NOT edit it** — its charter says of the evidence instruments *"Call them; never edit them."*
- **Fixed by the orchestrator at the merge, docstring only**, because the sentence became false *as a direct result of a merge the orchestrator performed*. That makes it integration responsibility, not a future card's inheritance.

### The mechanism

`docs/pwml_recovery_sprint/evidence/tree_pin.py:3-4` listed three premises for why the guard is needed:

> *"The venv's editable `.pth` names the primary checkout's `src`, **`pytest.ini` sets no `pythonpath`** and there is no `conftest.py` (F-003)"*

C-070 added `pythonpath = src`. The middle clause became measurably false the moment the merge landed.

### Why it was worth acting on rather than filing

This is the **F-087 class** — a comment asserting a measurement that has since drifted — but in the one module whose entire job is to refuse a run that cannot prove which tree it measured. **A guard whose stated rationale is false invites the next reader to conclude the guard is obsolete.**

### The function is unchanged, and that was verified rather than assumed

REV-070 measured the guard after the change, four ways. Re-confirmed by the orchestrator post-merge:

```
normal run                          2 passed, 1 skipped, exit 0
foreign PYTHONPATH                  violations: T2PW_FROM_WRONG_TREE
                                    REFUSED before collection. No test was run. Exit 98.
PYTHONPATH unset                    violations: T2PW_UNIMPORTABLE            Exit 98.
selection outside --expect-tree     violations: SELECTION_OUTSIDE_EXPECTED_TREE  Exit 98.
```

### The part that inverts the finding

**`pythonpath = src` moves plain-`pytest` resolution in the SAFE direction**, which is the opposite of what a reader of the stale sentence would assume. Measured by REV-070 with plain `python -m pytest`, same tree, foreign `PYTHONPATH`:

```
ini ON : t2pw.__file__ = C:\t\c070\src\t2pw\__init__.py          <- rootdir's own tree
ini OFF: t2pw.__file__ = C:\t\rev070base\src\t2pw\__init__.py    <- FOREIGN tree
```

Without the ini, plain pytest silently imported `t2pw` from whatever `PYTHONPATH` named. **The change enforces the very property `tree_pin.py` exists to enforce.** Under `pinned_pytest` it is a no-op either way, because `tree_pin.resolve_facts` binds `t2pw` in `sys.modules` before `import pytest` and `check` refuses a mismatch first.

**No hash covers `tree_pin.py`** (`bounded_run.py` hashes only itself), so a docstring edit changes no recorded identity. Verified before editing.

---

## F-089 — a test shells out to `git ls-files` at import time, so it cannot be collected in an exported base tree

- **Severity** LOW · **Registered 2026-08-21**, integration `09f7156` · **UNOWNED, deliberately not fixed**
- **Surfaced by REV-070** as the explanation for a discrepancy it refused to wave away: its own base isolated-collection sweep reported **22** failures where C-070's reported **21**.

### The mechanism

`tests/test_c030_canonical_identity_fallback.py:88` invokes `git ls-files` at **module import** time, so collection requires a working `.git`. `c045b_base_tree.py:35-38`'s `PATHSPEC` **excludes `.git`** from exported base trees.

So the file collects fine in a real checkout and fails in an export — and the two sweeps disagreed by exactly one file for exactly that reason. **The 21 are a strict subset of the 22 and all reproduce.**

### Why it is registered rather than fixed

It is not a defect in C-070's diff and it breaks nothing today. But it has one concrete consequence worth writing down: **anyone running `T2PW_ISOLATED_COLLECT_ALL=1` on an exported tree gets one spurious failure**, and a spurious failure in a sweep whose whole value is "0 failed" is how a real failure later gets dismissed.

### Remedy direction, for whoever owns it

Either defer the `git ls-files` call out of import scope into the test body, or have the sweep's docstring name this file as a known export-only failure. **The first is better** — the second is a comment that will drift, which is F-087 and F-088 both.

**A sentence in C-070's sweep docstring is the cheap interim.** It should ride along with the next card that owns either file rather than becoming a card of its own.

---

## F-090 — `bounded_run.py`'s descendant enumeration and RULE 5's record cap are in direct conflict for any high-fan-out job

- **Severity** MEDIUM (infrastructure, **live evidence-gate hazard**) · **Registered 2026-08-21**, integration `4c18736`
- **Surfaced by C-071**, whose base-tree materialisation produced a **149,703-byte** report that could not pass `check`. **The card disclosed the deletion rather than quietly renumbering**, which is the only reason this is visible at all.
- **Not C-071's defect**, and not fixable by any card that trips over it.

### The mechanism

`docs/pwml_recovery_sprint/evidence/g11/g11_evidence.py:74-77`:

```python
#: A structured record, never a log dump. 64 KiB is ~40x the size of a real
MAX_REPORT_BYTES = 65_536
```

and `:257` — `if size > MAX_REPORT_BYTES:  # RULE 5 (size): a record, not a log dump`.

`bounded_run.py` writes **every observed descendant PID and image name** into its report. For an ordinary pytest job that is a handful of entries. For a job that spawns thousands of short-lived children it is a log dump — which is exactly what RULE 5 exists to exclude, arriving through a field RULE 5 does not police.

`c045b_base_tree.py --rev <sha>` re-hashes every blob in the tree through `git cat-file`, so it spawns roughly **3820** transient processes. The resulting report was 149,703 bytes: **2.3× the cap**, composed almost entirely of dead PIDs.

**The job itself was clean** — exit 0, `FINAL SURVIVING COUNT: 0`, `cleanup: success`, and *"VERIFIED: every exported file re-hashes to its blob sha"*. **A compliant job produced a non-compliant record.**

### Why it cannot be worked around by the card that hits it

Re-running changes nothing: the descendant count is a property of `git cat-file`, not of how the job is invoked. The only options available to an implementer are

1. commit a report that fails `check` — turning the whole-tree gate red for everyone; or
2. delete it and disclose — spending D-025's contiguity evidence.

C-071 took (2) and said so. **Both are bad, and the choice should not exist.**

### Why this is worth registering rather than absorbing

**Merge rule 11 makes the G11 gate a hard merge condition**, and D-025 reads a contiguous sequence as proof nothing was deleted to fit a budget. This defect forces a gap into that sequence for a legitimate, correctly-run job — **so the very signal used to detect evidence tampering is consumed by a tooling limitation.** The next card that materialises a base tree hits it identically, with no warning in any charter.

### Remedy direction, for whoever owns it — UNOWNED

The narrowest fix is in `bounded_run.py`'s report writer, not in the cap: **cap or summarise the descendant list** — a count plus the first N entries, or a count plus a hash of the full list — rather than serialising every PID. That preserves what the field is for (proving nothing survived, which is a **count** question) while removing the unbounded growth.

**Do not raise `MAX_REPORT_BYTES` instead.** The comment at `:74` is right that a record is not a log dump, and raising the cap treats the symptom while leaving the field unbounded.

### The reusable lesson

**A size cap on a record does not bound a record whose fields are unbounded.** RULE 5 polices the artefact; nothing polices the field that grows with the child's behaviour. **A guard expressed as a limit on the output needs a matching limit on each input that can grow without bound**, or it fails exactly when the job is unusual — which is when its evidence matters most.

---

## F-091 — a serialized constant still tells consumers the semantic gating set is closed at four

- **Severity** LOW, but **it is the only one of its family that ships**, and it becomes wrong the moment C-071 merges · **Registered 2026-08-21**, integration `4c18736` · **UNOWNED**
- **Reported by C-071, correctly not fixed** — its charter granted `release_status.py` as *"one appended literal. Nothing else in this file."* The card obeyed the boundary and reported the prose it wanted changed, which is what a charter-bounded card is supposed to do.

### The mechanism

`src/t2pw/pipeline/release_status.py:72-75` defines `SEMANTIC_NO_GATING_CHECK_EVALUABLE`, whose text states that the gating set **is closed at four**. C-071 takes `SEMANTIC_GATING_CHECKS` to **five**.

**This one is not a comment.** It is a **serialized constant** — `semantic_verdict` returns it as the `reason` at `release_status.py:420` when no gating check is evaluable, and it reaches artifacts and operator-facing output. So the moment C-071 merges, the pipeline can emit a statement about its own configuration that is false.

### The rest of the family, which are comments and do not ship

* `release_status.py:77` — *"closed at exactly four"*
* `release_status.py:91` — *"equals the four constants"*
* `bench/acceptance.py:235` — read-only for C-071
* `tests/test_c056b_semantic_denominators.py:13` — module docstring

### Remedy

**Drop the number rather than change it to five.** *"…closed at four…"* → *"…closed…"*, with the count derived from `len(SEMANTIC_GATING_CHECKS)` where a count is genuinely needed. A constant that hard-codes the cardinality of a tuple it describes will drift again on the sixth addition, and the test that forces deliberate addition already guarantees a sixth will one day come.

### Why it is registered separately from the comment family

**PACK 11 RULING 4** says rationale prose carries an assertion's evidentiary burden and is worse than a stale assertion because no test goes red when it drifts. **This entry is worse still: it drifts *into an artifact*.** A stale comment misleads a maintainer reading the source; a stale serialized constant misleads a reviewer reading a run's output, who has no reason to suspect the string and no way to check it from the artifact alone.

**It must land with or before C-071's merge, not after.**

### Operational addendum to F-090 — `c045b_base_tree.py` needs ~182 s and will die under a 120 s agent tool clock

Measured by C-071, and worth stating because it cost a reservation before it was understood.

The base-tree export takes **182.48 s** for 3820 blobs. C-071's first attempt was killed at **120 s by its
own agent tool's wall clock** — `Exit code 143 — Command timed out after 2m 0s` — **not** by
`bounded_run.py --timeout`. The wrapper therefore never reached its report write and never promoted the
reservation, leaving an untouched three-key `g11_reserved` stub in `.staging/`.

**Nothing leaked:** the Windows Job Object's `KILL_ON_JOB_CLOSE` terminated the children when the shell
died, and the card verified zero survivors afterwards.

**Two things follow.**

1. **A charter that asks an agent to materialise a base tree must budget the tool clock, not just
   `--timeout`.** The wrapper's timeout is irrelevant if the caller's clock is shorter. C-071's second
   attempt used `--timeout 540` under a 580 s tool clock and completed.
2. **An abandoned reservation is the CORRECT residue of a killed job, not corruption.** `.staging/` is
   gitignored (`.gitignore:66`), so it can never reach a commit, and the standing rule is to leave a
   staging record alone rather than tidy it. C-071 left it, disclosed it, and separately identified two
   further staging entries as belonging to the orchestrator's concurrent Chunk D run and left those alone
   too — which is exactly the required behaviour.

### F-091 addendum — measured: NOTHING pins the literal text, so the fix is prose-only

Measured by the orchestrator at `7de755c`, because it decides whether this needs a card or can ride
along with one.

**The only assertion on the constant is by SYMBOL, not by value.**
`tests/test_semantic_release_gating.py:428` reads `assert reason == SEMANTIC_NO_GATING_CHECK_EVALUABLE`,
importing the name at `:47`. **Changing the string's value breaks no test.**

The complete family, measured — one ships, three do not:

| site | kind | ships? |
|---|---|---|
| `release_status.py:72-75` | the constant's own text, returned as `reason` at `:421` | **YES** |
| `release_status.py:77` | comment above `SEMANTIC_GATING_CHECKS` | no |
| `bench/acceptance.py:235` | docstring | no |
| `tests/test_c056b_semantic_denominators.py:13` | module docstring | no |

**So the whole remedy is four prose edits with no behavioural surface beyond the emitted string, and no
test pins that string.**

**It is still a patch to production source and will NOT be applied by the orchestrator.** F-088 was fixed
in place because it was a docstring in an *evidence instrument* that a merge had just falsified.
`release_status.py` and `acceptance.py` are production. *"Behaviourally inert"* would be a claim about an
unreviewed edit made by the person making it — which is precisely the failure PACK 11 RULING 4 was written
about, and applying it unreviewed would be a worse error than the stale string.

**Routing: fold into C-071 as an orchestrator-initiated scope addition after its review returns**, since
C-071 already owns `release_status.py` for one appended literal and its merge is held on Decision 5 anyway,
so there is no time pressure. The four sites are named above; the recommended text drops the number rather
than changing it to five.

---

## F-092 — two identical wall-clock timeouts recorded two different terminal reasons, and neither was `operation_timeout`

- **Severity** **HIGH** · **Registered 2026-08-21** from the **T-101 live run**, `runs_verify/2026-08-21_1822`, integration `839f529`
- **This is what T-101 was for.** The milestone's third acceptance clause is *"`budget_exhausted` distinct from failure"*, and measuring it found the seam is inconsistent with itself.
- **Not a model artifact.** Both legs are the same paper (PMC12444477), same run, same model, same 1800 s per-leg timeout. **The only difference is where the timeout was detected.**

### The measurement

```
--- PMC12444477 strict
   status               'timeout'
   failure_kind         'timeout'
   termination_reason   'budget_exhausted'
   stage                'unknown'
   message              'the child process was still running after 1800s and was killed,
                         so this paper+mode produced nothing (budget_exhausted)'

--- PMC12444477 research
   status               'timeout'
   failure_kind         'timeout'
   (termination_reason  ABSENT -- the key is not present at all)
   stage                'input'
   message              'extraction did not finish inside the time budget'
```

And, across the entire run directory:

```
grep -ric "operation_timeout"  ->  no hits anywhere
```

### Three distinct defects, one seam

**1. `operation_timeout` is never used, on a run containing two wall-clock timeouts.**
D-005 names `operation_timeout` as a termination reason and `OPERATIONAL_TERMINATION_REASONS`
is exactly `{budget_exhausted, operation_timeout}`. **A child killed after exceeding its wall
clock is the paradigm case for `operation_timeout`, and it was not used once.**

**2. A wall-clock kill is labelled `budget_exhausted`, which is a different fact.**
D-024 is explicit that a configured ceiling and an exhausted resource are different events, and
that is the whole reason `attempt_cap_reached` exists. **A leg that ran out of CLOCK is not a leg
that ran out of BUDGET.** Labelling the outer child-kill `budget_exhausted` collapses precisely
the distinction D-005 and D-024 exist to preserve — and it corrupts any denominator built on
`OPERATIONAL_TERMINATION_REASONS`, because both members now mean "we stopped it".

**3. The inner deadline path records NO terminal reason at all.**
This is the **F-070 defect class** — a leg that reliably terminates and says nothing — reappearing
on the extraction deadline path. C-064 closed it for the RAG round cap. **It is still open here.**

**The two paths disagree because they are two paths.** The batch runner's outer kill synthesises
`budget_exhausted` from outside the pipeline; the extraction deadline fires inside and its reason
never reaches the manifest. Neither consults the vocabulary D-005 defines.

### T-101 acceptance clause 1 is VIOLATED, and the phrase is hard-coded

The clause is *"no leg reports 'produced nothing'"*. `PMC12444477/strict` reports exactly that,
and the string is **not** an inference from an empty artifact set — it is **hard-coded into the
runner's timeout message**:

> *"…was killed, so this paper+mode **produced nothing** (budget_exhausted)"*

So the acceptance criterion cannot pass for **any** outer-killed leg, regardless of what that leg
actually produced. **The criterion is measuring a string literal, not a state.** Whoever owns the
fix must decide whether the clause means *"no leg emits this phrase"* or *"no leg silently yields
an empty artifact set"* — they are different tests and only the second is worth having.

### Remedy direction — UNOWNED, and it is not a one-liner

* The outer child-kill path should record **`operation_timeout`**, not `budget_exhausted`.
* The inner extraction-deadline path should record a terminal reason **at all**, and the same one.
* The runner's timeout message should stop asserting *"produced nothing"* as a fact about output
  it never inspected — the strict leg wrote `extraction_boundary_report.json` and
  `stage0_attempts.json` before it was killed, so *"produced nothing"* is **false as written**.

**⚠ Do not fix this by widening `OPERATIONAL_TERMINATION_REASONS`.** D-024 kept
`attempt_cap_reached` out of that set deliberately, and D-057 kept `round_cap_reached` out for
the same reason. The set is not the problem; the two paths not using it correctly are.

### What is NOT claimed

**Clause 2 of T-101's acceptance — *"`identical_empty_response` recorded where two draws share a
hash"* — was NOT exercised.** `grep -ric identical_empty_response` over the run returns nothing,
because no two draws shared a hash on these three papers. **That is not a pass and not a failure;
it is unexercised**, and T-101 cannot discharge that clause on this evidence. Recording it as
"clean" would be exactly the vacuous-pass failure mode RULING 13 exists to prevent.

---

## F-093 — the T-101 topics scope for PMC12312563 was recovered from the wrong field, and the scope guard caught it

- **Severity** LOW (control-plane / measurement input, **self-inflicted, corrected**) · **Registered 2026-08-21**
- **Registered against the orchestrator**, not against any card. Recorded because the *mechanism*
  of the mistake is reusable and will bite the next person who reconstructs a lost scope.

### What happened

`topics_t101.txt` was authored this session with
`PMC12312563 | menaquinone biosynthesis | Bacillus subtilis`. The live run returned:

```
Stage 0 read organism 'Listeria monocytogenes' but the batch requested 'Bacillus subtilis'
status: scope_conflict     both legs
```

**The pathway was right. Only the organism was wrong.**

### Why it was wrong, and this is the reusable part

PMC12312563's scope is recorded in **no topics file** — `topics_verify_subset.txt:13` lists the id
as a **Stage-0 scope abort**. It was reconstructed from the one historically successful run, whose
directory is:

```
runs/2026-07-27_1623/papers/PMC12312563__structures-of-listeria-monocytogenes-mend-in-th/
```

**The slug says `listeria-monocytogenes`.** The organism was nevertheless taken from that run's
`00_PAPER.txt` `organism:` field, which read `Bacillus subtilis`.

**That field records the REQUEST that was made, not the paper's subject.** So the reconstruction
faithfully recovered a *prior request* — and the prior request is exactly the one already on
record as producing a Stage-0 scope abort. **The evidence that the recovered value fails was
sitting in the same file it was recovered from.**

Confirmed against `01_source_text.txt` (47,976 bytes): *Listeria* appears, *Bacillus subtilis*
appears once as a homolog aside. Stage 0's reading is correct.

### The lesson

**When reconstructing a lost input, prefer a field the system DERIVED over a field a human
SUPPLIED.** A run directory's slug is derived from the paper; `00_PAPER.txt`'s `organism:` is
echoed from the request. Both were in front of the orchestrator; the derived one was right and the
supplied one was wrong, and they were quoted **two lines apart in the same authored file**.

### The guard worked, and that is the other half

`scope_conflict` refused to process the paper as though it were about an organism it is not,
wrote `scope_conflict.json` naming both the requested and observed values, and cost 8m49s + 4m41s
rather than a whole leg. **A bad input produced a precise refusal instead of a plausible wrong
answer.** That is the seam behaving exactly as designed and is worth recording as a positive.

### Disposition

**Corrected in `topics_t101.txt`** with the reasoning inline, and the two legs re-run from
`topics_t101_rerun_pmc12312563.txt` so the other four are not re-executed. **No code defect. No
card.**

---

## F-094 — PMC12452463/strict shipped `release_ready`, and the product contract says that outcome is never permitted

- **Severity** HIGH · **Class `product_contract_violation`** · **Registered 2026-08-22 from T-104**
- Evidence: `runs_verify/2026-08-21_2239`, leg 17/20; `evidence/t104_acceptance_report.{txt,json}`

### The measurement

```
PMC12452463  strict : PASS
   release : release_ready [pipeline ran; strict gates passed; semantic evaluation passed]
   payload=final_mapped.json  reactions=5  connected_core=2
   support: attribution=0% (0/5 reactions)  recall=0% (0/3 paper-stated reactions)
   enzyme recall: 2/3 (67%)   missing: EntE
   errors: missing pathway anchors=1, missing supported reactions=3, quarantined processes=2
```

### Independently corroborated by the artifact name

`PRODUCT_CONTRACT.md` §13 fixes the naming: *"`pathway.pwml` = `release_ready` only.
`pathway.review_required.pwml` = valid, needs review."* The committed strict artifacts are:

```
PMC12096016    pathway.review_required.pwml
PMC12180156    pathway.review_required.pwml
PMC12452463    pathway.pwml                  <-- the one paper the contract forbids to reach this
PMC12782028    pathway.review_required.pwml
PMC12856317    pathway.review_required.pwml
```

PMC12452463 is the **only** strict leg in the run that emitted a bare `pathway.pwml`. The violation
is therefore visible in the shipped artifact set itself, not only in the release field.

### Why it is a contract violation and not a benchmark disagreement

`PRODUCT_CONTRACT.md` §13 "Standing policy positions" states for this exact paper:

> Gold `export_rationale` records the route as chemically **broken** (EntA absent; nothing converts
> 2,3-dihydro-2,3-dihydroxybenzoate onward). Correct outcome after the index fix is
> `review_required` with `strict_acceptance_eligible=false`. **Never strict success.**

The contract outranks any test, benchmark result or inference from the code. `release_ready` is
strict success. No interpretation of the run makes this compliant.

The supporting numbers agree with the gold rather than with the release decision: **0% attribution
(0/5) and 0% recall (0/3)** of paper-stated reactions, `EntE` missing, 2 quarantined processes. The
pipeline shipped as release-ready a pathway in which not one retained reaction is attributable to a
paper-stated signature.

### This answers the open question carried out of T-103, and the answer is "yes, it mattered"

T-103 recorded that PMC12452463/strict reached the contractually required `review_required` **via
the semantic gate** (`actor_named_in_its_own_cited_span`, C-071), not via the route the gold
rationale describes, while coverage still passed (`minimum_core_satisfied: True` despite
`missing_anchors: ["EntA","Fur"]`, completeness 0.857). The open question was whether the gap
between the required outcome and the reason for producing it was load-bearing.

**It was.** In T-104 the semantic gate did not fire and nothing else held the leg: it went past
`review_required` to `release_ready`. The required outcome had been resting entirely on one gate,
and the coverage route that the gold calls broken still reports satisfied.

### Honest limit on the comparison

T-103 ran with `T2PW_SPECIES_LLM=0` and `T2PW_OFFLINE_CURATOR=1`; T-104 ran with both live, as an
RC requires. **The two legs are therefore not a clean A/B, and this must not be quoted as a
"regression from T-103".** It does not need to be: the violation is absolute against the contract
at T-104 alone. Per the standing trap, a single-leg change is not called a regression without a
re-run.

**Unowned. No card opened — this session was scoped to runs, triage and recording.**

---

## F-095 — the three gold organism traps abort as `scope_conflict`, and the gold expects two of them to export

- **Severity** HIGH · **Class `policy_disagreement`** · **Registered 2026-08-22 from T-104**
- **Needs a product-owner decision. Explicitly NOT a licence to change code.**

### The measurement

`PMC12657337`, `PMC12421875` and `PMC12312563` carry `requested_organism: "Bacillus subtilis"` in
the pinned gold set while the papers are *E. coli*, *L. lactis* and *L. monocytogenes*. All six
legs ended `SCOPE_CONFLICT` / `boundary=scope_ambiguity`. Stage 0 read the organism correctly every
time.

The pipeline nevertheless extracted good payloads before refusing:

| paper | gold `export` | gold `min_connected` | reactions | connected core | enzyme recall | metabolite recall |
|---|---|---|---|---|---|---|
| PMC12657337 | `strict_exportable` | 3 | 5 | **4** | 3/3 (100%) | 6/6 (100%) |
| PMC12421875 | `strict_exportable` | 7 | 11 | **10** | 8/8 (100%) | 10/10 (100%) |

Both **exceed** their gold connected-core floor with full enzyme and metabolite recall, and both
pass `organism_compatible` — no retained reaction is attributed to a forbidden organism.

### Why this is a policy disagreement and not a defect on either side

Both behaviours are deliberate:

* **The gold set** built these as traps on purpose — three `relevance_note`s say "ORGANISM TRAP" in
  capitals, and each case lists `Bacillus subtilis` in `forbidden_organisms`. Its scored property is
  the exported label: *"Every exported reaction must carry organism Escherichia coli; labelling them
  Bacillus subtilis, the requested organism, is a failure."* That test presupposes an export.
* **The pipeline** refuses on purpose — `config.py:194` `eligibility_stage0_conflict_aborts = True`,
  *"A Stage-0 reading that contradicts the batch request stops that run."*

`PRODUCT_CONTRACT.md` contains **no clause governing a requested-vs-observed organism conflict**, so
the contract does not settle it. Under the §14 adjudication rule only a `product_contract_violation`
justifies code, therefore **no code change is authorized by this finding.**

### Why it is the highest-leverage open item anyway

The scorer's own blocker ranking makes `scope_conflict` the **sole** blocker for both papers:

```
STRICT_EXPORT   1. scope_conflict  expected benefit: 2 paper(s) released outright
                   sole blocker for: PMC12421875, PMC12657337
RESEARCH_DELIV  1. scope_conflict  expected benefit: 3 paper(s) released outright
                   sole blocker for: PMC12312563, PMC12421875, PMC12657337
```

Strict PWML success is **0/4**; two of those four are these papers. This one policy question governs
half the strict denominator.

### The decision the product owner has to take

1. **Refuse is correct** — then the gold set's `expected_export` for PMC12657337 and PMC12421875 is
   wrong and the gold needs amending; the strict denominator drops from 4 to 2.
2. **Extract-and-relabel is correct** — then `eligibility_stage0_conflict_aborts` needs a scoped
   change so a paper whose organism is *read correctly* is exported under the **observed** organism,
   with the request recorded. This is the reading the gold's `export_rationale` assumes.

**Do not resolve this by editing the topics file.** Supplying the actual organism removes the trap
by handing the pipeline the answer and makes `forbidden_organisms` unexercisable.

### Disposition — RESOLVED by D-062 (2026-08-22)

The product owner took **neither** of the two readings above. **D-062** rules that a Stage-0
organism conflict whose reading is *correct* must **preserve the pathway as `review_required`
carrying the OBSERVED organism**, with the requested scope recorded alongside — it neither exports
strict under the wrong request nor drops the run.

The decisive point was merge rule 7: *"preserves incomplete-but-correct pathways as
`review_required` rather than dropping them."* Folding these legs to `STATUS_INELIGIBLE` — whose
definition reads *"not even a run: nothing was attempted, so nothing failed"* — was false on the
evidence, since both papers cleared the gold's own connected-core floor before being discarded.

**The Stage-0 guard is not weakened; only the disposition of a correct reading changes.** This does
not raise the strict PWML rate — `review_required` is not a strict export — and it does not ratify
the gold's `expected_export: strict_exportable` for PMC12657337 and PMC12421875, which D-062 leaves
open for reconciliation when the implementing card is written. No card was opened in this session.

### Correction to the record F-093 left

F-093 concluded that PMC12312563's `Bacillus subtilis` request was simply wrong because "its scope
is recorded in no topics file". That is true of *topics* files; **the pinned gold set has recorded it
all along as a deliberate override**, and `bench_acceptance.py --verify-plan` ratifies it as
`[pinned_override]`. F-093's "prefer the derived field" lesson remains correct for *reconstructing a
lost* scope and is not withdrawn; it does not govern a scope that is pinned and verifiable. The
T-101 re-run it prompted stands on its own terms.

---

## F-096 — seven false real identifiers were emitted on legs the pipeline reported as PASS

- **Severity** HIGH · **Class `product_contract_violation`** · **Registered 2026-08-22 from T-104**
- Acceptance priority 1 is declared absolute: "any non-zero count fails them however good the rest
  looks." Observed: **7**, papers PMC12180156, PMC12782028, PMC12856317.

### What was emitted, verbatim from the scorer

| paper / leg | entity | forbidden kind | accessions attached |
|---|---|---|---|
| PMC12856317 strict + research | `Pyridoxal 5'-phosphate` | `cofactor_as_protein` | drugbank DB00114, hmdb HMDB0001491, kegg C00018, chebi 18405, pubchem 1051 |
| PMC12180156 research | `succinyl-CoA` | `placeholder_product` | hmdb HMDB0001022, kegg C00091, chebi 15380, pubchem 439161 |
| PMC12782028 research | `SREBF1`, `SREBF2` | `regulator_as_metabolite` | uniprot P36956, Q12772 |
| PMC12782028 research | `LIPA`, `LBR` | `heading_or_prose` | uniprot P38571, Q14739 |

Gold reasons, quoted:

* PLP — *"The ALAS2 cofactor. Never a substrate, never a product, never a protein."*
* succinyl-CoA — *"**HALLUCINATION TEST**: zero occurrences in the entire 67,304-character file, body
  and references alike. The paper names ALAS2 without ever naming its substrates or its product, so
  any of these is fabrication."*
* SREBF1/2 — *"Transcription factors that appear in the Reactome cholesterol-biosynthesis gene list
  but catalyse nothing. The list is a membership set from enrichment statistics, not a reaction
  participant set."*
* LIPA — *"A degradative lysosomal acid lipase... Directionally the opposite of biosynthesis."*

### Why this is the worst class in the set

`goldset.py`'s own design note: *"Emitting one of these carrying a real external accession is the
single worst outcome the pipeline can produce, because every structural gate passes and the result
is silently wrong."*

That is exactly what happened — **every one of these legs was reported `PASS`.** PMC12782028's
research leg additionally reports *"20 verified / 0 Unknown-backed / 0 unresolved, of 20 protein
row(s); every protein row carries a real external identity"*, which reads as a perfect identity
result while four of those rows are enrichment-list membership entities that catalyse nothing.

`PMC12180156/research` is the sharpest case: a metabolite that **appears nowhere in the source text**
was emitted with four real database accessions.

A second-order defect on the same paper: `drugbank:db00114` is claimed by **two differently-named
entities**, `ALAS2` and `Pyridoxal 5'-phosphate` (`accession_claimed_by_multiple_entities`).

### Scope limit on the count

The report's own caveat: 8 legs were scored from `merged_payload.json`, which is pre-mapping, so
their identity counts are **floors, not measurements**. All four rows above were scored from
`final_mapped.json` and are real measurements. **The true total may be higher than 7; it cannot be
lower.**

**Unowned. No card opened.**

---

## F-097 — the batch runner calls the gold-specified-correct negative-control outcome a research-mode defect

- **Severity** LOW (reporting / triage doctrine, no wrong artifact shipped) · **Class
  `policy_disagreement`** · **Registered 2026-08-22 from T-104**

`SUMMARY.txt` files PMC13231680 under **"!! RESEARCH-MODE DEFECT !!"**, `class=broken`, above the
standing text *"Research mode is fail-open by design... therefore ANY research failure is a code
defect, not a data problem. Fix these before anything else."*

Both its legs ended `FAIL (no_reactions)` — *"the pipeline produced a pathway with no reactions and
no transports"*.

The gold set declares this paper a **negative control** (`mechanistic_relevance: context_only`) and
its `export_rationale` states: *"Nothing lipid-A-related is exportable at any level of partiality.
**The correct pipeline outcome is an empty pathway plus a rejection reason.**"*

An empty pathway plus a rejection reason is precisely what the pipeline produced. The acceptance
scorer agrees and excludes the paper as a declared negative control. **The blanket doctrine "ANY
research failure is a code defect" is wrong for a paper the gold set defines as one that must
produce nothing**, and it points the reader at "fix these before anything else". Left unqualified it
invites a change that would break a correct behaviour.

Note the runner is not wrong about the *other* paper in that section: PMC12444477's double timeout is
a real defect.

**Unowned. Reporting-side only — no pipeline change implied.**

---

## F-092 re-confirmed at T-104 (not a new finding)

Both PMC12444477 legs timed out at exactly 1800 s and both recorded:

> *"the child process was still running after 1800s and was killed, so this paper+mode produced
> nothing (**budget_exhausted**)"*

The hard-coded phrase *"produced nothing"* and the terminal reason `budget_exhausted` rather than
`operation_timeout` are exactly the F-092 defect, now observed on a third and fourth leg. **Recorded,
not fixed** — F-092 is open and unowned, and a milestone run is not the place to change the runner.

Separately, PMC12444477 is the sole extraction blocker in the run: it is `relevance=core` and neither
mode produced any payload, which is why extraction success is 7/8 rather than 8/8.

---

## F-098 - the "floor, not measurement" caveat on priority 1 is dischargeable by evidence, not by a card

- **Severity** LOW (measurement doctrine; no wrong artifact shipped) - **Class `policy_disagreement`**
- **Registered 2026-08-22 by the orchestrator during the T-104 correction phase**
- Evidence: `evidence/g11/C-073/01`-`06`; artifact census over `runs_verify/2026-08-21_2239`

### What the acceptance report recommended

*"Run the batch with the updated driver so `final_mapped.json` is persisted on failure paths."*
The stated purpose is to convert the 8 legs scored from pre-mapping `merged_payload.json` from
floors into measurements, making priority 1 trustworthy.

### Why that card should not be written

An artifact census over all 20 T-104 legs:

| legs | `final_mapped.json` | `merged_payload.json` | outcome |
|---|---|---|---|
| 10 PASS | present | present | scored from `final_mapped.json` |
| 6 `scope_conflict` + 2 `no_reactions` | **absent** | present | scored from `merged_payload.json` |
| 2 TIMEOUT | absent | absent | no payload at all |

`merged_payload.json` carries **zero** `mapped_ids` on *every* leg, PASS legs included
(`PMC12856317/strict`: 0 in `merged_payload.json`, 12 in `final_mapped.json`). That is
`driver.py:1248-1253` working as documented, not a defect.

On the 8 failure legs the mapping stage **never executed** - Stage 0 aborted on the organism
conflict, or the pathway had no reactions to map. `final_mapped.json` is therefore not
*unpersisted*; it does not exist, because nothing produced it. A driver change cannot persist an
object that was never built.

### The consequence for the count

**7 is a measurement over every leg on which an external accession can exist**, not a floor in any
sense that matters. A leg that never reached mapping cannot emit a false real identifier. The
caveat is literally true and practically empty.

### What is actually worth changing, and it is reporting-side only

`bench/acceptance.py:930` / `:1042-1046` emit the same "floor" caveat whether the leg reached
mapping and lost the artifact, or never reached mapping at all. Those are different facts and only
the first is a measurement gap. The scorer should distinguish
`mapping_did_not_execute` from `mapped_artifact_missing`.

**Residual risk, stated honestly:** this holds for the three failure modes T-104 exhibited
(`scope_conflict` at Stage 0, `no_reactions`, TIMEOUT). A failure that aborts *after* mapping -
for example the gate-failure terminal path at `driver.py:1879` - would leave a genuine
measurement gap. No T-104 leg failed that way. If a T-105 leg does, this finding does not cover it.

**No pipeline card opened. Reporting-side only.**

---

## F-099 - withholding a PathBank scalar is not durable to the strict PWML export against a live DB

- **Severity** LOW (one row in the whole committed corpus; does not touch the accessions priority 1
  counts) - **Class `product_contract_violation`**
- **Registered 2026-08-22 by the orchestrator during the C-073 correction phase**
- Disclosed by the C-073 implementer as outside its ownership boundary; verified independently.

### The mechanism

C-073's identity admission withholds an unsupported or kind-conflicting identifier by moving it
into `mapping_meta.rejected_mapped_ids` and clearing it from `mapped_ids` and its scalar column.
For a compound that includes the `pathbank_compound_id` scalar.

`src/t2pw/pwml/compound_resolution.py:503-504`:

```python
legacy_id = _db_id(row, ["pathbank_compound_id", "pw_compound_id", "pathwhiz_id"])
if legacy_id is not None:
    fallback = dict(row)
    fallback["db_status"] = "legacy_id_unverified"
    ...
```

The legacy branch is taken **only when the row still carries a PathBank id**. With the id
withheld, `legacy_id` is `None`, the branch is skipped, and the row falls through to
`PathWhizCompoundResolver`, which resolves **by name**. Against a reachable PathBank the resolver
can therefore re-attach the very record the identity gate just refused, pre-freeze.

The refusal is a statement about the ENTITY - "this stage cannot support this identity". A
name-keyed re-lookup downstream does not consult that statement, so the two disagree and the
later one wins.

### What is and is not affected

- **Not affected: `mapped_ids`.** hmdb / kegg / chebi / pubchem / cas / biocyc / chemspider stay
  withheld, and those are what F-096 and acceptance priority 1 actually count. `final_mapped.json`
  records the refusal correctly.
- **Affected: the strict PWML export path only, and only with a live PathBank.** The re-resolution
  is name-keyed, so it can only fire for a row whose name the DB recognises.
- **Blast radius, measured:** exactly one row in the 53-artifact committed corpus
  (`succinyl-CoA`, PMC12180156/research, withheld by Pass A). Pass B has never withheld a PathBank
  scalar - its only corpus hit is `drugbank`. And Pass A is dormant in production pending the
  wiring blocker, so **today this cannot fire at all.**

### Why it is recorded now rather than fixed

`pwml/compound_resolution.py` and `pipeline/prefreeze_resolution.py` are outside C-073's ownership
boundary, and the fix is not obviously narrow: the right shape is probably for the resolver to
consult `rejected_mapped_ids` before a name-keyed lookup, which is a change to a shared resolution
path used by every compound on every leg. That deserves its own card, its own corpus measurement
and its own review - not a widening of a card already in its second round.

It is also **not currently reachable**: the only pass that withholds a PathBank scalar is Pass A,
and Pass A cannot fire until the source-index wiring lands. **This finding becomes live at the
same moment that wiring does, and should be carded together with it.**

### Related

Same root shape as merge rule 8's concern - a later stage repairing what an earlier gate refused -
but on the identity axis rather than the biology axis, and pre-freeze rather than post-freeze.

---

## F-100 — the declared negative control produced reactions and shipped `release_ready`

- **Severity** HIGH · **Class `product_contract_violation`** · **Registered 2026-08-23 from T-105**
- Evidence: `runs_verify/2026-08-22_2147`, PMC13231680 both legs; `evidence/t105_acceptance_report.{txt,json}`

### The measurement

```
PMC13231680 [NEGATIVE CONTROL]   gold: relevance=context_only  export=partial_only  min_connected=0
  strict   : PASS   release : release_ready [pipeline ran; strict gates passed; semantic evaluation passed]
             payload=final_mapped.json  reactions=1  connected_core=1   artifact: pathway.pwml
             [FAIL] minimum_connected_core: negative control: 1 reaction(s) retained where the paper supports 0
  research : PASS   reactions=2
  errors: unsupported reactions=1 (strict), 2 (research)
```

The gold `export_rationale` is unambiguous: *"Nothing lipid-A-related is exportable at any level of
partiality. **The correct pipeline outcome is an empty pathway plus a rejection reason.**"*

At T-104 both legs produced exactly that — `FAIL (no_reactions)`, which F-097 defended as correct.
At T-105 the same paper produced reactions and its strict leg emitted a **bare `pathway.pwml`**,
which `PRODUCT_CONTRACT.md` §13 defines as `release_ready` — no human review needed — for a paper
the gold says supports nothing.

### This is priority 2 and it moved from PASS to FAIL

T-104: priority 2 `[PASS]` — 0 unsupported retained reactions.
T-105: priority 2 `[FAIL]` — **3**, all on this paper.

### It is NOT caused by C-072 or C-073, and C-072 could not have caught it

- C-072's cap only ever **demotes**; it cannot promote a leg to `release_ready`.
- The cap is gated on `verdict.declared`. This gold case declares **0 anchors**
  (`requested_pathway_anchors_present: 0/0`), so there is no declared core and the cap correctly
  abstains. That is the designed behaviour, not a miss.
- C-073 withholds identifiers; it neither creates nor retains reactions.

The cause is Stage-1 draw variance: this draw extracted reactions where the T-104 draw extracted
none. Per the standing trap, that is not called a deterministic regression on one observation —
but the **contract outcome is wrong whenever it happens**, and the run recorded it happening.

### What the pipeline lacks

There is no production predicate that prevents a pathway from reaching `release_ready` when the
requested-core declaration is empty. `evaluate_core_coverage` treats an undeclared core as
unjudgeable (`completeness` is `None` by design, `release_status.py:188`), and `classify_release_status`
falls through to `RELEASE_READY`. "Nothing was asked for" is currently read as "nothing is missing".

---

## F-101 — a one-reaction pathway shipped `release_ready` on a paper the gold says cannot export strict

- **Severity** HIGH · **Class `product_contract_violation`** · **Registered 2026-08-23 from T-105**
- Same family as F-094, reached by a route C-072 cannot see.

### The measurement

```
PMC12856317   gold: relevance=partial  export=partial_only  min_connected=1
  strict : PASS   release : release_ready [pipeline ran; strict gates passed; semantic evaluation passed]
           payload=final_mapped.json  reactions=1  connected_core=1   artifact: pathway.pwml
           [ok] requested_pathway_anchors_present: 4/4 requested-pathway anchors present
           [ok] minimum_connected_core: largest chemically connected core is 1; this case requires 1
```

Gold `export_rationale`: *"A single reaction cannot constitute an exportable multi-step pathway.
Emitting a strict heme biosynthesis pathway from this paper requires importing seven steps the
paper never mentions."*

### Why C-072 does not and should not fire here

All four requested anchors matched, so `missing_anchors` is empty and the incomplete-core cap
correctly abstains. C-072 fixed the mechanism F-094 identified — a declared core with **unmatched**
anchors — and this leg is not that mechanism. **The card did its job; the class is wider than the
card.**

### The tension the gold itself carries

The gold sets `min_connected: 1` for this case while its own `export_rationale` says one reaction
is not exportable. The pipeline satisfies the floor and ships. Either the floor or the rationale
has to give, and **that is a product-owner decision, not an orchestrator one.** Recorded, not taken.

### Net effect on the run

T-104 emitted **one** bare `pathway.pwml` (PMC12452463 — the F-094 violation, now fixed).
T-105 emits **two** (PMC12856317, PMC13231680). **The bare-PWML count went up, not down.**
Nobody should read T-105 as an improvement on this axis.

---

## F-102 — the acceptance scorer flags exactly the accession rule C-073's review rejected

- **Severity** MEDIUM · **Class `policy_disagreement`** · **Registered 2026-08-23 from T-105**
- Needs a product-owner decision. Explicitly NOT a licence to change code.

### The measurement

Every surviving `accession_claimed_by_multiple_entities` finding at T-105 is **within-kind**:

```
uniprot:P0ADI4   <- EntB / holo-EntB              (both entities.proteins)
uniprot:P10378   <- EntE / enterobactin synthase  (both entities.proteins)
```

`bench/semantic.py:908-919` counts any accession answering to two differently-**named** rows as a
conflict. `no_real_id_or_name_conflict` is a **gating** semantic check (`release_status.py:101`),
so these demote real legs.

### Why this is a disagreement and not a defect on one side

C-073 was **rejected in review** for using precisely this predicate in the pipeline, because
**D-035 clause 3c** rules that a matching stable external identifier is *proof two differently-named
rows are the same biological entity*. The corrected pipeline rule refuses only **cross-kind** claims.
Measured over 53 committed artifacts, the name-difference rule strips 92 claimant-incidences across
36 rows of which 41 of 42 pairs are legitimate; the kind rule strips 2 rows, all target.

So the pipeline now follows D-035 and **the scorer does not**. `EntE` / `enterobactin synthase`
is the clearest case — those are the same protein under symbol and full enzyme name, exactly the
shape clause 3c was written for.

### The second-order problem: the gold disagrees with D-035 too

`holo-EntB` carrying `uniprot:P0ADI4` is scored as a **false real identifier**
(`forbidden_kind: strain_or_construct`), i.e. the gold says the holo form must not claim EntB's
accession. D-035 clause 3c says a shared stable identifier proves they are the same entity.
**Both cannot be right.** This is not a scorer bug and not a gold typo — it is an unresolved
question about whether a modified form of a protein is the same biological entity as the protein.

### What must NOT be done

Do not "fix" this by reverting C-073's predicate — that was reviewed and rejected on measured
evidence. Do not edit the gold to make a number move. The decision is which of D-035 clause 3c and
the gold's `forbidden_identity` list governs a holo/apo pair, and it belongs to the product owner.

---

## F-103 — an unstated request produces a satisfied coverage verdict

- **Severity** HIGH · **Class `product_contract_violation`** · **Registered 2026-08-23 from T-105**
- **Root cause of F-100.** Owned by **C-074 arm B**.

### The measurement

`runs_verify/2026-08-22_2147`, `PMC13231680/strict` — the declared negative control, requested as
**lipid A biosynthesis in Escherichia coli**:

```
coverage.requested_core_source : pathway_context
coverage.requested_context     : {"pathway_name": "",          <-- EMPTY
                                  "likely_organism": "Escherichia coli",
                                  "key_compounds": ["phthalylsulfacetamide (PSA)","meropenem (MEM)",
                                                    "Zn2+","sulfacetamide"],
                                  "key_proteins":  ["NDM-1 (...)","LpxC (...)"]}
coverage.requested_core_declared : True
coverage.coverage_ratio          : 1.0
coverage.unmatched_terms         : []
release.status                   : release_ready
```

The requested pathway **never reached the context**. `pathway_name` is the empty string, and the
key terms are the paper's own subject matter. Coverage then scored 6/6 against those terms,
reported `completeness: 1.0`, and the leg shipped a bare `pathway.pwml`.

### Why this defeats the check

`collect_requested_core_terms` (`strict_quarantine.py:753`) documents the hazard itself:

> *"Never derived from the surviving graph. Terms taken from what survived would match whatever
> survived, which is not a test."*

The terms here are not literally taken from the surviving graph — they come from a **Stage-0
reading of the same paper**, which defeats the check by the identical mechanism one step earlier.
The docstring's stated assumption, that `key_compounds` / `key_proteins` are *"what the preprocessor
produced when it read the request, before any extraction could bias them"*, does not hold when
Stage 0 reads the paper to build the context.

Contrast `PMC12856317/strict` in the same run, where `pathway_name` is `"heme biosynthesis"` and
the key terms are genuine heme-pathway members. **The field is right when Stage 0 states a pathway
and degenerate when it does not** — so the defect is specifically the unstated case, not the
context mechanism as a whole.

### The rule that is missing

`evaluate_core_coverage` has no notion of "the request was never stated". `requested_core_declared`
is True whenever *any* terms were collected, regardless of whether a pathway was named.
"Nothing was asked for" is currently read as "nothing is missing".

### Scope note

This finding is about the **release decision** only. Whether Stage 0 *should* propagate the batch's
requested pathway into `pathway_context` is a separate question about the acquisition seam, and it
is **not** carded here — C-074 makes the unstated case unjudgeable rather than satisfied, which is
correct regardless of how the context comes to be empty.

---

## F-104 — a job with many short-lived children can never produce a compliant G11 report

- **Severity** LOW (infrastructure / evidence discipline; no wrong artifact shipped) ·
  **Class `policy_disagreement`**
- **Registered 2026-08-23**, surfaced by the C-075 implementer.

### The measurement

`bounded_run.py` records **every** descendant PID it observes into its JSON report.
`g11_evidence.py` caps a report at 64 KiB. The base-tree export helper `c045b_base_tree.py` spawns
**1,745 short-lived `git.exe` children**, producing a 172,974-byte report that `check` then fails
as `report_too_large`.

The run itself was clean: `exit code (real) : 0`, `FINAL SURVIVING COUNT : 0`,
`cleanup : success`, duration 224.84 s.

### Why it is recorded rather than worked around

Editing a G11 report is forbidden, so the implementer deleted the oversized artifact rather than
commit a non-compliant one, and said so. That is the correct call under the current rules and it is
the reason this finding exists rather than a silently trimmed file.

**This is the F-090 hazard actually biting.** F-090 (MEDIUM, open) predicted exactly this and noted
it *"did not bite at T-104 (45 descendants fit)"* and *"only threatens base-tree exports"*. It has
now bitten, on a base-tree export, as predicted.

### The disagreement to settle

The descendant list is written for survivor accounting, and the lifecycle rule that matters is
`FINAL SURVIVING COUNT : 0` — a number, not the list. A report that records 1,745 dead PIDs proves
nothing the count does not. Candidate resolutions, none taken here:

1. record descendants as a **count plus the survivors only**, keeping full detail only when
   survivors are non-zero;
2. raise or remove the 64 KiB cap for reports whose survivor count is zero;
3. accept that base-tree exports are exempt and record them out of band.

All three change evidence policy, which is a product-owner call. **No card opened.** Until it is
settled, any job spawning thousands of children should be expected to fail `check` even when the
run is perfectly clean, and that failure must not be read as a lifecycle violation.

---

## F-105 — the source index rides into the interactive curator prompt at a second, unfixed site

- **Severity** MEDIUM (cost and prompt quality; no wrong artifact shipped) ·
  **Class `product_contract_violation`**
- **Registered 2026-08-23**, surfaced by the C-075 reviewer. **Blocks arming the app in anger.**

### The measurement

C-075 Extension A fixed the audit LLM prompt by filtering at serialization
(`audit_json_llm.payload_for_prompt`, keyed off `identity_admission.SOURCE_INDEX_KEY`). Verified on
a real leg: index blob 64,880 bytes, prompt byte-identical with and without the key present,
`source_text_index` absent from the prompt, and the payload the audit returns still carries the
index so `map_payload` still refuses `succinyl-CoA`.

**There is a second site with the same shape and it is not fixed.**

`src/t2pw/curation/interactive_curator.py:164`
`strip_payload_for_interactive_context` is a **blacklist** — `_RAW_TEXT_KEYS` (exact membership)
and `_BULKY_KEY_TOKENS` (substring). `source_text_index` matches neither: the exact set carries
`source_text`, not `source_text_index`, and no bulky token is a substring of it.

`run_interactive_curator_round` (`streamlit_app.py:2804`) then `json.dumps` that payload into a
multimodal prompt (`interactive_curator.py:255-259`). Once the source index is armed, roughly
65 KB of index rides into **every interactive curator round**.

### Why it is registered rather than fixed

`interactive_curator.py` is outside C-075's ownership boundary, and the C-075 implementer
correctly declined to reach for it. Widening a card mid-correction to a file it does not own is
how boundaries stop meaning anything.

### The design point worth keeping

Extension A and this site fail differently for the same reason: **one filters by an allow-list
keyed off the constant, the other by a blacklist that has to be remembered.** A blacklist silently
admits every future key. If this is carded, prefer making the interactive path consult the same
`PROMPT_OMITTED_PAYLOAD_KEYS` constant rather than adding one more string to a list nobody will
revisit.

### Scope

**Not currently reachable.** The wiring hunk that arms the source index is not merged at the time
of registration, so no production path writes the index yet. This finding becomes live at exactly
the moment that hunk lands, and should be closed **before** the interactive app is used against a
real paper — the batch legs that T-106 runs do not touch the interactive curator.

---

## F-106 — the seed paper's provenance mark carries a PATHWAY NAME where a paper title belongs

- **Severity** LOW (curator-facing reporting only; no wrong artifact ships, no gate reads it) ·
  **Class `reporting_defect`**
- **Registered 2026-08-23**, surfaced by the C-075 reviewer. **Does NOT block T-106.**
- **C-075's verdict is unaffected** — proved below, not asserted.

### The observation

Two corpus rows carry a `rag_provenance` record whose `source_title` is a *pathway name*:

```
runs/2026-07-27_1623/.../PMC12312563__structures-of-listeria.../strict   entity "α-ketoglutarate"
  rag_provenance = {"source_id": "seed_paper",
                    "source_title": "menaquinone biosynthesis",          <-- a PATHWAY
                    "source_type": "paper", "chunk_id": ""}
  the real paper title sits one field away, in the row's own source_papers:
    {"source_id": "PMC12312563", "title": "Structures of Listeria monocytogenes MenD in ThDP-bound
     and in-crystallo captured intermediate I-bound forms."}

runs/2026-07-28_0919/.../PMC13278307__an-overview-of-mobile-colistin.../strict  entity "pmrCAB operon"
  rag_provenance = {"source_id": "seed_paper",
                    "source_title": "lipid A modification (colistin resistance)",   <-- a PATHWAY
                    "source_type": "paper", "section": "introduction",
                    "chunk_id": "cbea424a0ce3ebb83a7dacb98484ae4c"}
```

**The premise needs one correction before anything else.** The three fields are not mutually
inconsistent: `source_type: "paper"` is TRUE (the seed *is* a paper) and `source_id: "seed_paper"`
is TRUE (`provenance.py:102` `SEED_SOURCE_ID`). **Exactly one field is wrong — `source_title`.**
That matters for scoping: this is one mislabeled string, not a fabricated provenance record.

### Where the mislabel originates — two adjacent lines, both in the app

The pathway name reaches `source_title` by two routes, and the two observed rows exercise one each.

**Chunk route** (produced the `pmrCAB operon` row — it carries a `chunk_id`):

`src/t2pw/app/streamlit_app.py:590`
```python
seed=ingest.seed_candidate(
    seed_text,
    title=str(_safe_dict(pathway_context).get("pathway_name") or ""),   # <-- HERE
```
becomes `CandidatePaper.title` (`ingest.py:288`, whose honest fallback `"uploaded seed paper"` is
reached only when `pathway_name` is empty), then every seed `Chunk.source_title` with
`source_type="paper"` hardcoded (`ingest.py:344-345` and `375-376`) and `source_id=SEED_SOURCE_ID`
(`ingest.py:286`), then `rag_provenance` via `synthesize._provenance_from_chunk` (`synthesize.py:411-417`).

**Descriptor route** (produced the `α-ketoglutarate` row — `chunk_id` is `""`):

`src/t2pw/app/streamlit_app.py:637`, inside the literal at `:633-640`
```python
_seed_name = str(pathway_context.get("pathway_name") or pathway_context.get("title") or "").strip()
...
"source_id": "seed_paper",
"source_title": _seed_name or "uploaded seed paper",    # <-- HERE
"source_type": "paper",
```
consumed by `synthesize._seed_source_descriptor` (`synthesize.py:446-470`) and stamped as the
fallback provenance at `synthesize.py:946-947`.

**Why a pathway name is what lands there.** Stage 0's context schema has **no document-title field
at all** — `src/t2pw/llm/prompts/preprocess_system.txt:151-155` and `:189`, where `pathway_name` is
specified as *"short name of the pathway (e.g., \"obafluorin biosynthesis\")"*. So
`streamlit_app.py:630`'s `pathway_context.get("title")` fallback can never fire, and the only
non-empty string available at that seam is the pathway name. The app is not mislabeling by
accident so much as reaching for a field that does not exist.

### It cannot change admission. Measured, not read.

`provenance_route` (`identity_admission.py:504-510`) tests **key presence and non-emptiness only**:

```python
for key in PROVENANCE_ROUTE_KEYS:
    value = row.get(key)
    if isinstance(value, dict) and value:
        return key
```

Replayed offline over the two committed artifacts against their own `01_source_text.txt`
(`evidence/g11/F-106/03-replay-identity-support-r3.json`, exit 0, `FINAL SURVIVING COUNT : 0`,
`cleanup : success`) — only the `rag_provenance` field values vary between variants:

```
as-committed                 -> not_evaluated  identity_from_another_admitted_source  route=rag_provenance
source_title="" type=pathbank-> not_evaluated  identity_from_another_admitted_source  route=rag_provenance
fabricated id AND title      -> not_evaluated  identity_from_another_admitted_source  route=rag_provenance
{"whatever": 1}  (no source fields at all)
                             -> not_evaluated  identity_from_another_admitted_source  route=rag_provenance
rag_provenance = {}          -> unsupported    identity_not_supported_by_source       route=-
rag_provenance DELETED       -> not_evaluated  identity_from_another_admitted_source  route=source_papers
```

Two things follow. **The values are inert** — a wholly fabricated `source_id`/`source_title`/
`source_type` behaves identically to the committed one, and to a dict with no source fields at all.
**The mislabeled tuple is not even the load-bearing route for these two rows**: delete
`rag_provenance` outright and both still route on `source_papers`, which names a real PMC id
(`PMC12312563`, `PMC12844150`).

And the route never grants anything. `identity_admission.py:583-586` returns
`STATUS_NOT_EVALUATED`, never `STATUS_SUPPORTED`; `map_ids._admit_identities` pass A
(`map_ids.py:8198-8213`) `continue`s on both `supported` and `not_evaluated` and calls
`_withhold_identity` **only** on `unsupported`. The pass can withhold. It cannot admit.

The committed suite already pins this and did so before this finding existed:
`tests/test_c075_source_support_armed.py:1249-1283`
(`test_the_route_value_is_never_read_so_seed_paper_is_not_a_special_case`) asserts *via the AST*
that `SELF_REFERENTIAL_ROUTE_SOURCE` has zero `Load` occurrences, and `:1286+`
(`test_an_abstaining_row_acquires_nothing_and_is_never_called_supported`) pins
`supported == 0`, `not_evaluated == 3`, and pass B still stripping the kind conflict.

### The real escape hatch, stated honestly — and it is not this

A genuinely unsupported row *is* rescued by **the presence of any non-empty route carrier**.
Probe on the F-096 hallucination (`runs/2026-08-02_2130/papers/PMC12180156/strict`,
`evidence/g11/F-106/04-fabricated-mark-probe.json`, exit 0, `FINAL SURVIVING COUNT : 0`,
`cleanup : success`):

```
succinyl-CoA (8 accessions), as-committed  -> unsupported   <-- correctly refused
  + rag_provenance {"source_id":"seed_paper","source_title":"menaquinone biosynthesis", ...}
                                            -> not_evaluated
  + rag_provenance {"x": 1}                 -> not_evaluated   <-- SAME
```

`{"x": 1}` is as effective as the full fabricated tuple. So the admission surface is *carrier
presence*, which is the product owner's 2026-08-23 ruling implemented as written, measured by
C-075 at 32 rescues / 0 collateral / 3 hallucinations still refused over 678 eligible rows.
**Correcting `source_title` would not narrow that surface by a single row.** Anyone who wants that
surface narrowed is asking for a different decision from the product owner, not for this defect.

### The two rows are sound, and their synonym evidence never touches the mark

Both identities were resolved by the mapper from its own candidates, recorded in `mapping_meta`:

| row | resolver evidence | paper's own spelling (measured) |
|---|---|---|
| `α-ketoglutarate` | `source: "db"`, `providers: ["PathBankDB"]`, one candidate `pathbank_compound_id 134` *Oxoglutaric acid* (`short_name` AKG), score 1.0, `chosen_rule: pathbank_compound_id`, `confidence: 1.0` → 9 accessions incl. `HMDB0000208` | `2-oxoglutarate` **8×**, `ketoglutarate` **0×** |
| `pmrCAB operon` | `provider: "UniProt"`, `source: "api"`, `P30843` *Transcriptional regulatory protein BasR*, `gene_names: ["basR","pmrA"]`, E. coli K12, `reviewed: true`, score 0.85, `alias_source: "gene_name"`, `matched_alias: "pmrA"` | `PmrAB` **4×**, `pmrCAB` **0×** |

Neither resolution consulted `rag_provenance`. `rag/conform.py:283-292` reaches the same verdict
independently, from an audit of every `merged_payload.json` under `runs/`: it names
`'pmrCAB operon' (PMC13278307)` and `'α-ketoglutarate' (PMC12312563)` among seed-tagged rows that
are the ONLY row for their species, and refuses to gate on the tag because *"a tag-based rule would
have deleted real chemistry"*.

**One precision correction to C-075's own prose, which is not an evidence failure.**
`identity_admission.py:242-243` and `tests/test_c075_source_support_armed.py:1226-1228` state that
PMC13278307 *"writes `pmrA` 4 times"*. Measured: all four occurrences are inside **`PmrAB`** (raw
case-insensitive `pmrA` = 4, raw `PmrAB` = 4; under the module's own `normalize_text`, token
`pmra` = **0** and `pmrab` = **4**). The substance holds exactly — the paper discusses the PmrAB
regulator throughout and never writes `pmrCAB` — but the count is of a substring, not a standalone
token. Recorded here so that a future reader who re-measures `pmra` = 0 does not mistake an
imprecise count for a fabricated citation.

### Where the wrong title actually surfaces

Only in prose a human reads:

- `rag/tiers.py:275` — `cited = ", ".join(s["title"] or s["source_id"] for s in quoted[:3])` builds
  the tier **reason** string, so a curator can be told `"menaquinone biosynthesis"` is one of the
  *"identified papers"*. The tier itself is decided on `source_id` counts (`tiers.py:273`
  `independent = len({s["source_id"] for s in quoted})`) and never on the title;
  `source_type` is not even in tiers' `rag_provenance` carrier map (`tiers.py:100-103`).
- `pipeline/lineage.py:113` — `source_type` is declared *"advisory FREE text, not a closed
  vocabulary: never aggregate on it"*, so by its own contract it drives nothing.
- `rag/provenance.py:168-177` (`_has_resolvable_source`), `bench/semantic.py:476`,
  `pipeline/strict_quarantine.py:2386` — all three test **non-emptiness or presence** of
  `source_id` / the carrier. None reads `source_title`; none reads `source_type`.

That is the whole blast radius: one misleading sentence in a curator-facing rationale.

### Classification and the T-106 verdict

`reporting_defect`. The sprint's rule is that only `product_contract_violation` justifies a code
change, and nothing here violates the contract — no artifact is wrong, no gate consumes the field,
no identifier is admitted or withheld differently, and the two affected rows are correct biology
with database- and resolver-grounded identities.

**It does not block T-106.** No release gate and no acceptance scorer reads `source_title` or
`source_type`, so fixing it cannot move a T-106 number in either direction; and both origin lines
sit in `streamlit_app.py`, the file the whole RC runs through, where a cosmetic edit before a
20-leg run is pure downside. **No card is opened.** If it is ever fixed, fix it as a rider on a
card that already owns `streamlit_app.py:588-592` and `:633-640` — pass the seed document's own
title where one exists and otherwise let `ingest.py:288`'s existing `"uploaded seed paper"`
fallback stand — and only after T-106.

### C-075 is not affected

Its trusted-route clause does not rest on trusting the mark's **values**. It rests on the mark's
**presence**, the value is never read (pinned by AST assertion), the route yields `not_evaluated`
and never `supported`, the pass can only withhold, and both affected rows would route identically
on `source_papers` with the seed record deleted. C-075's measured numbers — 32 route rescues,
4 span rescues, 3 hallucinations still refused, 0 collateral over 678 eligible rows — are numbers
about carrier presence and are untouched by this finding. **No re-review is owed.**

---

## F-099 — AMENDMENT, 2026-08-23: re-measured after the pass was armed. Severity was wrong, and it blocks T-106

- **Severity revised LOW → HIGH.** Class unchanged: **`product_contract_violation`**.
- **Amended by the Lead Orchestrator, 2026-08-23**, at integration tip `9831fc1`.
- **Carded as C-078** (`prompts/C-078.md`). Evidence: `evidence/g11/F-099/01-blastradius.json`,
  `02-seamdetail.json`, `03-dblive.json` — all three `FINAL SURVIVING COUNT : 0`, `cleanup : success`.

### Why it was re-measured

The original entry deferred the finding on the grounds that *"Pass A is dormant in production
pending the wiring blocker, so **today this cannot fire at all**."* That blocker is discharged: the
wiring hunk landed at `f12115a` and C-075 merged at `81b0bf9`. The pass is armed. The finding
became reachable at that moment and its recorded blast radius had never been re-derived against the
committed corpus.

Re-deriving it changed the answer in two independent ways. Both are measured offline from committed
artifacts and current source.

### Correction 1 — the scope note is wrong. `mapped_ids` IS affected

The original entry states: *"Not affected: `mapped_ids`. hmdb / kegg / chebi / pubchem / cas /
biocyc / chemspider stay withheld… Affected: the strict PWML export path only."*

`db_resolver.py:459-472`, the admitted branch of `apply_compound_db_resolution`, ends:

```python
out["hmdb_id"]     = chosen.get("hmdb_id")
out["kegg_id"]     = chosen.get("kegg_id")
out["pubchem_cid"] = chosen.get("pubchem_cid")
out["chebi_id"]    = chosen.get("chebi_id")
for key in ["description", "cas", "biocyc_id", "chemspider_id", "drugbank_id"]:
    out[key] = chosen.get(key)
out["mapped_ids"] = _mapped_ids_from_row(chosen)      # wholesale overwrite
```

`mapped_ids` is **rebuilt from the DB row** — not merged, not filtered. Every namespace the identity
gate withheld returns if the matched PathBank record carries it, and those namespaces are exactly
what acceptance **priority 1** counts.

This runs **pre-freeze**, so the restored identity enters the canonical graph. That is not an
exporter repairing biology after the freeze (merge rule 8); it is the refused identity becoming
canonical.

### Correction 2 — the corpus figure is 5, not 1

The original *"exactly one row in the 53-artifact committed corpus (`succinyl-CoA`,
PMC12180156/research, withheld by Pass A)"* was measured on a Pass-A-only corpus. Against the 22
committed `final_mapped.json` artifacts of T-104 and T-105:

```
rows carrying mapping_meta               : 379
rows with non-empty rejected_mapped_ids  : 11
  of those, in a compounds container     :  8
  of those, refusal names a pathbank id  :  5   <- the exact F-099 seam
```

All five are `2,3-dihydro-2,3-dihydroxybenzoate`, each with `chebi`, `kegg`, `pubchem` and
`pathbank_compound_id` refused and the PathBank scalar genuinely cleared from the row — which is
precisely the precondition that skips the legacy branch at `compound_resolution.py:502` and drops
the row into the name-keyed resolver:

| leg | rejected namespaces | pathbank scalar still on row |
|---|---|---|
| PMC12096016/research | chebi, kegg, pathbank_compound_id, pubchem | False |
| PMC12096016/research | chebi, kegg, pathbank_compound_id, pubchem | False |
| PMC12096016/strict | chebi, kegg, pathbank_compound_id, pubchem | False |
| PMC12452463/strict | chebi, kegg, pathbank_compound_id, pubchem | False |
| PMC12452463/strict | chebi, kegg, pathbank_compound_id, pubchem | False |

The mechanism discriminates correctly, which is the evidence this reading is real rather than a
pattern match: `Fe3+` (PMC12096016/research) has only `kegg` refused, **still carries its PathBank
scalar**, and therefore takes the legacy branch untouched. The five that fall through are exactly
the five whose scalar is gone.

Two of the affected legs are load-bearing. **PMC12452463/strict** is F-094's leg — the only strict
leg that emitted a bare `pathway.pwml`. **PMC12096016** is C-076's leg.

### The live-DB precondition holds

F-099 is conditional on a reachable PathBank. All 11 `db_resolution` records in the committed corpus
read `available: True` with **no** unavailability reason. The DB was reachable on both T-104 and
T-105 and is reachable now.

### Also confirmed

`grep -rn "rejected_mapped_ids" src/ --include=*.py` returns hits in `mapping/identity_admission.py`,
`mapping/map_ids.py` and `pipeline/entity_identity.py`, and **zero hits anywhere in
`src/t2pw/pwml/`**. The refusal record is not consulted at this seam at all.

### Open, deliberately not carded here

Three refused rows — `Fur`, `apo-EntB`, `holo-EntB` on PMC12096016/research — carry a refused
`uniprot` and are proteins. `pwml/` has an admission gate for compounds only; `_admit_db_identity`
has no protein counterpart. **Whether the protein path has the same defect is a separate question**
and C-078 is instructed to report on it without changing it.

### The one premise C-078 must prove before it may implement

Everything above shows the re-attachment *can* happen. It does not establish that pre-freeze
compound resolution runs **after** the identity gate on a production leg. If the gate re-imposes its
refusal downstream, the finding is moot. C-078 §3 requires that ordering to be proven behaviourally
through a production entry point before any fix is written, and to be reported rather than worked
around if it does not hold.

---

## F-092 — AMENDMENT, 2026-08-23: re-measured at `5f8a230`. One defect of three survives, and it does not block T-106

- **Severity revised HIGH → MEDIUM.** The finding is now **split by defect**, because its three
  parts do not share a class and two of them do not justify code.
- **Amended by the measurement lane, 2026-08-23**, at integration tip `5f8a230`.
- Evidence: `evidence/g11/F-092/01-replay-classify.json` (deterministic replay of the stored
  T-104/T-105 rows through the current vocabulary and the current serializer),
  `02-pin-test.json` (`test_deadline_leg_timeout.py` + `test_batch_driver_seam_golden.py`,
  9 passed). Both `FINAL SURVIVING COUNT : 0`, `cleanup : success`.
- **No live leg was run and the timeout was not reproduced.** T-101, T-104 and T-105 recorded
  this seam on six legs between them; a seventh 30-minute kill would have bought nothing.
- **Still holds at `50420a1`.** The tip advanced twice during the re-measurement (`30b8fd6`
  LEDGER, `50420a1` C-079 charter). Both are docs-only — `git diff 5f8a230..50420a1 -- src/ tests/`
  is empty — and C-079's boundary is `src/t2pw/curation/**`, disjoint from everything below.

### The seam did not move, so nothing merged since could have closed it

The re-measurement was commissioned on the hypothesis that the deadline module landed after
F-092 was registered and may already have closed part of it. **It did not, and it could not.**

```
git log -1 --format=%cI 985355f   ->  2026-08-13T09:26:36-06:00   C-032: one monotonic per-leg deadline
git log -1 --format=%cI 839f529   ->  2026-08-21T17:43:16-06:00   the SHA F-092 was measured at
git merge-base --is-ancestor 985355f 839f529   ->  YES

git diff --stat 839f529 HEAD -- src/t2pw/batch/driver.py \
                                src/t2pw/batch/runner.py \
                                src/t2pw/pipeline/deadline.py
   ->  (empty)
```

C-032 shipped `deadline.py`, `runner._timeout_row`'s `classify_child_kill` call and
`driver._finalize_timeout`'s `classify_interaction_timeout` call **in one commit, eight days
before F-092 was written**. All three files are **byte-identical** between `839f529` and
`5f8a230`. F-092 was therefore measured against the current code from the start, and "the
deadline module may have closed it" is not available as an explanation for anything below.

### What T-104 and T-105 actually recorded

Four legs, two runs, one paper. The two runs disagree — and the disagreement is the finding.

| run | leg | path | `status` / `failure_kind` | `stage` | `seconds` | `termination_reason` | `operational_failure` | `budget` |
|---|---|---|---|---|---|---|---|---|
| T-104 `2026-08-21_2239` | strict | OUTER `runner._timeout_row` | `timeout` / `timeout` | `unknown` | 1800.26 | `budget_exhausted` | `True` | present |
| T-104 | research | OUTER | `timeout` / `timeout` | `unknown` | 1800.37 | `budget_exhausted` | `True` | present |
| T-105 `2026-08-22_2147` | strict | **INNER** `driver._finalize_timeout` | `timeout` / `timeout` | `stage1` | 1749.44 | **ABSENT** | **ABSENT** | **ABSENT** |
| T-105 | research | OUTER | `timeout` / `timeout` | `unknown` | 1800.30 | `budget_exhausted` | `True` | present |

Messages, verbatim:

```
OUTER (3 legs)  'the child process was still running after 1800s and was killed,
                 so this paper+mode produced nothing (budget_exhausted)'
INNER (T-105 strict)
                'audit and DB mapping did not finish inside the time budget'
                detail: 'AppTest script run timed out after 471.9942160999999(s)'
```

Each of the three OUTER rows carries a full `budget` record — `leg_timeout_seconds: 1800.0`,
`leg_timeout_default_seconds: 3600.0`, `leg_timeout_overridden: true`,
`child_deadline_seconds: 1680.0`, and a negative `remaining_seconds` (`-0.26`, `-0.37`, `-0.30`)
showing each child was killed just past its ceiling. The INNER row carries no `budget` key at all.

```
grep -ric "operation_timeout"  runs_verify/2026-08-22_2147/   ->  no hits anywhere
grep -ric "operation_timeout"  runs_verify/2026-08-21_2239/   ->  no hits anywhere
```

### Per-defect verdict

| # | Defect as registered | At `5f8a230` | Class | Justifies code? |
|---|---|---|---|---|
| 1 | `operation_timeout` never used on a run with wall-clock timeouts | **reproduces, but is not independent** — it is the *observable* of defect 3 | inherits defect 3 | no separate card |
| 2 | A wall-clock kill labelled `budget_exhausted` is the wrong fact | **REFUTED** | **`policy_disagreement`** | **no** |
| 3 | The inner deadline path records no terminal reason at all | **reproduces** | **`product_contract_violation`** | yes, after T-106 |
| — | T-101 clause 1 violated by a hard-coded `"produced nothing"` | string is real; **the premise that it is false is itself false** | `policy_disagreement` (acceptance wording) | no |

### Defect 2 is REFUTED, and implementing its remedy would be the violation

`PRODUCT_CONTRACT.md` § 9, which outranks any test, benchmark or inference from code, defines
the two reasons at `:260-261`:

| Reason | Means |
|---|---|
| `budget_exhausted` | another recovery step might have helped; **wall-clock did not allow it** |
| `operation_timeout` | **an individual external operation** exceeded its deadline |

**The contract equates `budget_exhausted` with the wall clock in its own definition.** D-005
does the same throughout — *"one monotonic per-leg deadline across all stages, recording elapsed
and remaining budget"*, and *"insufficient budget for the next rung → stop, record
`budget_exhausted`"*. In D-005 the per-leg clock **is** the budget; there is no second,
non-clock budget anywhere in the entry.

F-092's argument — *"a leg that ran out of CLOCK is not a leg that ran out of BUDGET"* — has no
support in either document. Its D-024 citation transposes an argument about the **attempt**
ceiling onto the **wall-clock** ceiling. D-024's excluded reason is `attempt_cap_reached`, and
the sentence F-092 leans on says the opposite of what it was cited for:

> *"D-005 calls the cap 'a safety ceiling, not a promise', which is a different fact from **a leg
> that ran out of clock**."*

D-024 uses *"a leg that ran out of clock"* as the thing `OPERATIONAL_TERMINATION_REASONS`
already covers, in order to exclude the attempt cap from it. It never says a clock-exhausted leg
should be `operation_timeout`.

The measured rows are arithmetically correct under that reading. Replayed through the current
classifier (`evidence/g11/F-092/01-replay-classify.json`):

```
classify_child_kill(elapsed=1800.26, leg_timeout=1800)  -> 'budget_exhausted'   == recorded
classify_child_kill(elapsed=1800.37, leg_timeout=1800)  -> 'budget_exhausted'   == recorded
classify_child_kill(elapsed=1800.30, leg_timeout=1800)  -> 'budget_exhausted'   == recorded
```

All three children ran to the ceiling. `deadline.classify_child_kill` (`deadline.py:553-566`) is
not a constant: `tests/test_deadline_leg_timeout.py:74-75` pins `_row(400.0)` →
`OPERATION_TIMEOUT`, so the OUTER path **does** emit `operation_timeout` — for a child killed
while leg budget remained, which is what the contract says that reason means, and which did not
happen on any of these six legs.

**Adopting F-092's remedy would relabel every ceiling kill as `operation_timeout`, contradict
`PRODUCT_CONTRACT.md:260`, contradict D-005's *"record `budget_exhausted`"*, and break a green
committed test.** Defect 2 is a `policy_disagreement` with the locked vocabulary, and the sprint
rule that only `product_contract_violation` justifies code applies directly. **No code.**

### Defect 3 REPRODUCES — and it is the entire explanation of defect 1

`driver._finalize_timeout` classifies correctly and then the row throws the answer away.

```
driver.py:1730   outcome.termination_reason        = classify_interaction_timeout(detail, explicit=reason)
driver.py:1731   outcome.termination_is_operational = is_operational(outcome.termination_reason)

driver.py:684-710  RunOutcome dataclass fields  -> neither name is declared
driver.py:746-781  RunOutcome.to_dict()         -> neither name is emitted
```

Behavioural replay at the tip, feeding back the **stored T-105 strict `detail` string**:

```
detail = 'AppTest script run timed out after 471.9942160999999(s)'
outcome.termination_reason         = 'operation_timeout'
outcome.termination_is_operational = True
to_dict() keys = [counts, detail, failure_kind, files, issue_codes, message,
                  mode, paper_id, seconds, stage, status, warnings]
'termination_reason' in row  = False
```

**`operation_timeout` WAS computed on the T-105 strict leg and was discarded by the serializer.**
That is the whole of defect 1: `operation_timeout` is absent from every run directory not because
the vocabulary is unused, but because the one path that produced it cannot write it down.

This violates `PRODUCT_CONTRACT.md` § 9 *"On timeout or budget exhaustion, preserve … elapsed and
remaining budget · the next recovery step that was skipped · **the exact stop reason**."* The
INNER row preserves neither the stop reason nor the budget. Class: **`product_contract_violation`**.

**It is deliberate, not an oversight, and it is actively pinned green.** `_finalize_timeout`'s own
docstring says so — *"Nothing is added to `RunOutcome.to_dict` — that would edit `RunOutcome`,
outside this card's boundary"* — and C-032 shipped a test that asserts the absence:

```python
# tests/test_deadline_leg_timeout.py:125-135
def test_a_driver_timeout_keeps_its_row_byte_identical() -> None:
    """Preservation: the classification is an attribute, never a manifest field.
    The golden driver diff hashes ``RunOutcome.to_dict()``; a new key here would
    move the pinned ``input_timeout`` leg."""
    row = _timed_out("app.run() timed out after 45s").to_dict()
    assert "termination_reason" not in row and "release_status" not in row
```

That file is 7/7 green at the tip. Any fix must **invert** this assertion as a declared baseline
move under merge rule 4, not quietly delete it.

**Nothing downstream reads the field.** `git grep termination_reason operational_failure --
src/t2pw/batch/report.py src/t2pw/bench` returns zero hits. The absence moves no acceptance
number, no denominator and no release status. It costs the manifest its § 9 completeness — which
is a real contract violation and worth one small card — but it costs T-106 nothing.

### The `"produced nothing"` string — the string is real, the objection to it is not

`src/t2pw/batch/runner.py:958` hard-codes the phrase, and it is emitted for every OUTER-killed leg
regardless of that leg's output. So T-101 acceptance clause 1 (*"no leg reports 'produced
nothing'"*) does grep a string literal rather than a state, and that observation stands.

**But F-092's grounds for calling the phrase false do not survive measurement.** The entry states
*"the strict leg wrote `extraction_boundary_report.json` and `stage0_attempts.json` before it was
killed, so 'produced nothing' is **false as written**."* An artifact census says otherwise:

```
runs_verify/2026-08-21_1822/papers/PMC12444477/   (T-101)  strict/RESULT.txt   research/RESULT.txt
runs_verify/2026-08-21_2239/papers/PMC12444477/   (T-104)  strict/RESULT.txt   research/RESULT.txt
runs_verify/2026-08-22_2147/papers/PMC12444477/   (T-105)  research/RESULT.txt
                                                           strict/  cleaning_report.json
                                                                    extraction_boundary_report.json
                                                                    merged_payload.json
                                                                    rag_admission_report.json
                                                                    stage0_attempts.json
                                                                    stage1_payload.json
```

Those two files exist **only at T-105**, and only for the **strict** leg — the one leg that took
the **INNER** path and therefore **never emits the phrase at all** (its message is *"audit and DB
mapping did not finish inside the time budget"*). Every leg that does emit *"produced nothing"*
carries `files: []` on its row and nothing but `RESULT.txt` on disk, because the parent never
receives the child's `outcome.artifacts`. **On all six observed legs the phrase is true as
written.**

**There is also no message/detail self-contradiction.** `message` says *"produced nothing"*;
`detail` says *"nothing about this paper was judged"*. Those agree — one is about artifacts, the
other about the biological verdict, and both are correct. There is nothing here to fix in
`runner.py`.

What remains is a **measurement-criterion** question that belongs to whoever writes T-106's
acceptance, not to the runner: clause 1 should assert *"no leg silently yields an empty artifact
set"* rather than grepping for a phrase. `policy_disagreement`. **No code.**

### T-106 verdict: DOES NOT BLOCK

1. The only surviving defect writes nothing any scorer reads — zero hits in `bench/**` and
   `batch/report.py`.
2. It cannot move a T-106 number in either direction, so fixing it first buys no measurement.
3. Its fix lands in `src/t2pw/batch/driver.py`, the file every leg of the RC runs through, and
   must re-baseline a SHA-256 golden. Editing that file and re-capturing that golden immediately
   before a 20-leg run is pure downside — the same reasoning F-106 applied to `streamlit_app.py`.
4. T-106 will re-run PMC12444477 under the same 1800 s ceiling and will observe the seam again for
   free. If a live confirmation of the INNER path is ever wanted, take it from there.

**No card is opened here.**

### If it is ever carded — the boundary, and why it is not disjoint today

**Function-level the fix is disjoint from C-077; file-level it is not.**

| Would change | Why |
|---|---|
| `src/t2pw/batch/driver.py` :: `RunOutcome` fields (`:684-710`) | declare `termination_reason: str = ""`, `termination_is_operational: bool = False` |
| `src/t2pw/batch/driver.py` :: `RunOutcome.to_dict` (`:746-781`) | emit both **conditionally**, in the existing `if self.scope_conflicts:` style |
| `tests/test_deadline_leg_timeout.py` (`:125-135`) | **invert** the pin, with the docstring rewritten to say why |
| `tests/test_batch_driver_seam_golden.py` :: `GOLDEN` | re-baseline **one slot**, `input_timeout` |

`src/t2pw/batch/driver.py` is on C-077's ownership table (`prompts/C-077.md:173`), and C-077 is
dispatched in a worktree at `129d9b2`. Its function boundary there is `_reconcile_stage0_scope`
only, so the two edits do not touch the same lines — but they are the same file on two live
branches, which is a merge collision the sprint sequences rather than races. `src/t2pw/batch/runner.py`
is explicitly **out** of C-077's bounds (`prompts/C-077.md:189`, *"owned by C-032"*), so the runner
side is free — and, per the section above, the runner side needs no change.

**Sequence it after C-077 merges and after T-106.** Nothing is lost by waiting.

**Deliberately excluded from that boundary:** the `budget` record. § 9 also asks for *"elapsed and
remaining budget"*, which the INNER row lacks, but `_finalize_timeout` has no `_Budget` in scope —
supplying one means a new parameter threaded through all five call sites (`driver.py:2009, 2043,
2069, 2171, 2394`). That is a wider change than the stop reason and should be judged separately.

**Do not widen `OPERATIONAL_TERMINATION_REASONS`, and do not touch `classify_child_kill`.**
F-092's original warning against the first is right and is reaffirmed. The second is new: the
OUTER classification is **correct** and the only remaining fix is a serialization gap on the INNER
path.

---

## F-107 — D-062's literal `review_required` is not constructible at the seam D-062 describes

- **Severity** MEDIUM · **Class `policy_disagreement`** — it needs a product ruling, not a code fix
- **Registered 2026-08-23 by the Lead Orchestrator**, surfaced by the C-077 implementer and
  independently verified before registration. **Does NOT block T-106.**
- **C-077 is not defective for this.** It delivered the honest disposition available at the seam and
  escalated the rest rather than fabricating a measurement.

### The ruling, and what the seam can actually support

**D-062** (LOCKED, 2026-08-22) says a Stage-0 organism conflict preserves the extracted pathway
**"as `review_required`, carrying the OBSERVED organism"**.

`review_required` cannot be constructed where the conflict is detected. Two independent reasons,
both verified against current source rather than taken from the implementer's report:

1. **`classify_release_status` pins the status before any `REVIEW_REQUIRED` branch is reachable.**
   `pipeline/release_status.py:641-660` is one `elif` chain, and its second arm is

   ```python
   elif not strict_gates_passed:
       status = DIAGNOSTIC_ONLY
       reasons.append(REASON_STRICT_GATES_BLOCKED)
   ```

   All five `status = REVIEW_REQUIRED` sites (`:651, :658, :694, :731, :818`) sit below it. The
   strict technical gates never ran at this seam — the conflict aborts at `driver.py:2130`, step 3b,
   before audit, DB mapping, freeze and export. Passing `strict_gates_passed=True` to reach
   `review_required` would **fabricate a measurement**, which is the exact defect class
   `_finalize_gate_failure`'s F-055 docstring exists to prevent.

2. **`PRODUCT_CONTRACT` §4 defines `review_required` as "Valid, useful PWML *produced*"**, and
   `ReleaseStatus.produced_pwml` encodes it. No PWML exists at this seam and none can. Measured: not
   one of the six committed T-105 conflict legs holds a `.pwml` of any name.

`PRODUCT_CONTRACT.md` **outranks any test, benchmark result or inference from the code**. It does
not obviously outrank a locked product ruling, and that collision is what this finding records.

### The third option, and why it was not taken in a card

Delivering D-062's literal wording requires driving the run onward — audit → DB mapping → freeze →
export — under an organism **known to be wrong**, emitting `pathway.review_required.pwml`
(`driver.py:1423`).

That is a **new product decision nobody has taken**. What an audit and a DB mapping should do when
the organism is known wrong is settled nowhere in the contract or the decision log. It is also the
one shape that could accidentally produce the strict export D-062 forbids outright. The Lead
Orchestrator is not authorised to take it, so C-077 was scoped to the disposition and this was
escalated.

### What C-077 shipped instead, and why it is the lesser inexactness

`diagnostic_only` with an explicit reason,
`stage0_scope_conflict_stopped_the_run_before_serialization`, plus `requested_scope` beside
`observed_context` on the manifest row.

Note this is **also** inexact: `PRODUCT_CONTRACT` §4's gloss on `diagnostic_only` is *"recovery and
retrieval could not establish a defensible pathway core"*, and that is untrue on these legs — at
T-104, PMC12421875 reached a connected core of 10 against the gold's own floor of 7, with 8/8
enzyme and 10/10 metabolite recall. The reason constant is worded to disclaim it, following the
precedent `_finalize_gate_failure` already set for reading `diagnostic_only` as *"a statement about
SERIALIZATION, not about the biology"*.

**Both candidate states are inexact. `diagnostic_only` is the one already precedented for a stopped
run and the only one that cannot fabricate a gate result.**

### The real gap

`PRODUCT_CONTRACT` §4 has **no state for "a defensible core was extracted but never serialized
because a scope guard correctly stopped the run."** D-062 assumed one existed. Until the product
owner rules, the conservative disposition stands.

### What a ruling would need to settle

1. Does the run continue to serialization under the OBSERVED organism, or does the record stay
   pre-serialization?
2. If it continues: what may the audit and the DB mapping do when the organism is known wrong, and
   what stops that path from ever reaching a strict export?
3. Or: does `PRODUCT_CONTRACT` §4 gain a fourth state for a defensible-but-unserialized core?
4. D-062's own explicitly-left-open question should be settled in the same ruling — the gold's
   `expected_export: strict_exportable` for `PMC12657337` and `PMC12421875` was never reconciled
   with the ruling, and neither paper counts as a strict-export success until it is.

### Why it does not block T-106

The harm D-062 names is a completed extraction being recorded as *"nothing was attempted"*
(`eligibility.py:123-131`, `report.py:76-81`). C-077 closes exactly that: the row now carries
`release_status.pipeline_executed: true`, which is the machine-readable refutation, plus both the
requested and observed scope in separate fields. The six T-106 conflict legs will be classified
truthfully whichever way this is later ruled. What remains open is **which** truthful classification
they get, not whether they get one.

---

## F-108 — the rejected within-kind accession rule also lives in the PRODUCTION release gate, not only the acceptance scorer

- **Severity** HIGH · **Class `product_contract_violation`**
- **Registered 2026-08-23 by the Lead Orchestrator**, disclosed by the C-076 implementer as outside
  its ownership boundary and **independently verified before registration**.
- **Widens F-102.** F-102 was registered as *"the pipeline now follows D-035; the scorer does not."*
  Measured, that is only half true: the **identity admission** layer follows D-035 after C-073, but
  the **semantic release gate** does not.
- **Carded as C-080.** Evidence: `evidence/g11/F-094/01-reopen-probe.json`, `02-reopen-probe2.json`.

### The second copy

`src/t2pw/bench/semantic_production.py:252-260`, inside `_audit_entities`:

```python
for (namespace, accession), names in sorted(holders.items()):
    distinct = sorted({_s.normalize_name(n) for n in names if n})
    if len(distinct) > 1:
        conflicts.append({
            "kind": "accession_claimed_by_multiple_entities", ...
            "reason": f"{namespace}:{accession} is claimed by {len(distinct)} differently-named entities",
        })
```

**No kind check of any sort.** Any accession answering to two differently-*named* rows is a
conflict — which is exactly the predicate C-073's review **rejected** as contradicting **D-035
clause 3c**, and exactly what the product owner's 2026-08-23 identity ruling forbids flagging.

This is not the benchmark scorer. It feeds `CheckResult(name=_s.CHECK_ID_CONFLICT, ...)`, and:

* `bench/semantic.py:98` — `CHECK_ID_CONFLICT = "no_real_id_or_name_conflict"`;
* `pipeline/release_status.py:114-119` — that name is a member of `SEMANTIC_GATING_CHECKS`;
* `pipeline/strict_quarantine.py:2424` imports `evaluate_production_semantics` — **the production
  release path**.

So it gates real runs. C-076 corrects `bench/semantic.py` only, so after C-076 the two seams
**disagree**: the acceptance scorer says these are not conflicts and the production gate still says
they are.

### It is demonstrably firing on real legs

From the committed `quarantine_report.json` files — `release.semantic_failed_checks`:

| run | leg | recorded status | failed gating checks |
|---|---|---|---|
| T-105 `2026-08-22_2147` | `PMC12096016/research` | `review_required` | `no_real_id_or_name_conflict`, `actor_named_in_its_own_cited_span` |
| T-105 `2026-08-22_2147` | `PMC12452463/strict` | `review_required` | **`no_real_id_or_name_conflict` — sole failing check** |
| T-104 `2026-08-21_2239` | `PMC12856317/research` | `review_required` | **`no_real_id_or_name_conflict` — sole failing check** |

`PMC12096016`'s collisions were measured by C-076 to be `uniprot:P0ADI4` ← `EntB` / `holo-EntB` and
`uniprot:P10378` ← `EntE` / `enterobactin synthase`, **all four claimants in `entities.proteins`** —
i.e. precisely the within-kind case the ruling says must stop being a conflict.

### Correcting it does NOT reopen F-094 — measured, because the risk was real

`PMC12452463/strict` is F-094's leg, and F-094 is a `PRODUCT_CONTRACT` §13 violation. Its recorded
coverage verdict is `minimum_core_satisfied: true`, `coverage_ratio: 0.785714`,
`surviving_processes: 9`, `reasons: []` — so the technical chain reaches `release_ready` and **only
the semantic cap demoted it**. That made "correcting this rule reopens F-094" a live hazard, and it
had to be settled before the card was written rather than after.

Settled by deterministic offline replay of `classify_release_status` using each leg's own recorded
coverage verdict and semantic fields, toggling only whether `no_real_id_or_name_conflict` is among
the failed checks. The replay reproduces all three recorded statuses exactly before the toggle,
which is what makes the counterfactual trustworthy:

| leg | as recorded | with the within-kind rule corrected | why it still holds |
|---|---|---|---|
| `PMC12452463/strict` | `review_required` | **`review_required`** | `requested_core_anchors_unmatched: 2,3-dihydroxybenzoate (DHB), EntA, Fur` — **C-072's rule, firing independently** |
| `PMC12096016/research` | `review_required` | **`review_required`** | `semantic_evaluation_failed: actor_named_in_its_own_cited_span` |
| `PMC12856317/research` | `review_required` | **`review_required`** | `requested_core_anchors_unmatched: hemin` |

**No leg flips to `release_ready`. `strict_acceptance_eligible` stays `False` on all three.**
F-094 stays closed by C-072's own mechanism, which is independent of the semantic cap. Correcting
this rule raises the strict PWML rate by **zero** on all 22 committed legs, so it cannot be a case
of weakening a gate to increase PWML production (merge rule 6).

### What must be predicted for T-106, not discovered

On T-106's draws a leg could fail **only** the within-kind check **and** have all its declared
anchors matched. Such a leg would legitimately become `release_ready` under the ruling. That is the
correct outcome, not a regression — but it must be predicted before the run and classified
deliberately afterwards, never quoted as an unexplained strict-rate improvement.

### Blocks T-106

**Yes.** It gates real legs on a rule the product owner has ruled invalid, and leaving it means
T-106's release path and its acceptance scorer measure two different things — which is precisely
the condition that made T-105 unquotable on this axis.

---

## F-109 — `TEST_MATRIX` § 0 rule 10 explicitly REFUSES `pythonpath = src`, and `pytest.ini` has carried it since C-070

- **Severity** MEDIUM · **Class `control_plane_contradiction`** — a merge-gate document forbids, in
  terms, something the tree it governs has been doing for many cards.
- **Registered 2026-08-23 by the Lead Orchestrator**, surfaced by the C-079 implementer and
  **verified directly before registration.** **Does NOT block T-106** — see the mitigation below.
- **Not chargeable to C-079 or to C-070.** Neither introduced a defect; the two were written
  without knowledge of each other.

### The contradiction, both halves quoted

`docs/pwml_recovery_sprint/TEST_MATRIX.md:101-110`, inside rule 10:

> **`PYTHONPATH` is not evidence and a printed path is not evidence.** The venv's editable `.pth`
> names the primary checkout's `src`, `pytest.ini` sets **no** `pythonpath` and there is no
> `conftest.py` […] `pytest.ini` **must not** gain `pythonpath = src`: it was considered as a remedy
> for F-003 and is **refused**, because pytest *prepends* those entries, so it would sit ahead of the
> `PYTHONPATH` pin and **make every base-tree G9 proof silently measure the tip** — the same defect
> class as F-003, aimed at the proofs themselves.

`pytest.ini:1-8` at the current tip:

```ini
[pytest]
testpaths = tests
# F-066 / C-070. Resolved relative to rootdir and prepended to sys.path at config time,
# before any collection. Without it a test file could only import t2pw when some file
# collected earlier in the same run had already inserted src itself, so 21 of the 156
# test files failed collection when run on their own while every multi-file chunk
# stayed green. Not an editable install: nothing here mutates .venv.
pythonpath = src
```

Added by **C-070** at `5bc600e` (*"a test file collects on its own, and no pinned count moves"*), for
a real and unrelated defect: 21 of 156 test files could not be collected individually. Its own
rationale comment is sound. It simply landed a remedy another document had already refused, and
neither side was updated.

### Why the stated hazard did NOT materialise

The prohibition's reasoning is about `sys.path` **ordering**: a prepended `pythonpath` sitting ahead
of a `PYTHONPATH` pin. Two things blunt it.

1. **`pythonpath` is resolved relative to `rootdir`, not to the primary checkout.** A pytest run
   whose `rootdir` is the base worktree resolves `src` to *that tree's own* `src`. The hazard is
   real only when pytest is invoked with a rootdir in one tree and the intent to measure another.
2. **Rule 10's own operative control is the resolved-path pin, not the prohibition.** The same rule
   says: *"Only the **resolved** path, compared against the expected tree and written to a committed
   verdict, settles which tree was measured."* That is `evidence/pinned_pytest.py` with
   `--expect-tree` / `--pin-verdict`, and it asserts the resolved `t2pw.__file__` rather than trusting
   any path variable. **It catches exactly the failure the prohibition was written to prevent.**

### Audited: every base-tree G9 proof of this session was pinned

| card | base proof | pin verdict |
|---|---|---|
| C-076 | reviewer's own base run | `t2pw` resolved in `C:\t\rv076b\src` |
| C-077 | implementer + reviewer, both sides | `tree=C:\t\c077base`, `violations: []` |
| C-078 | three base runs | pinned to `C:\t\c078base` / `C:\t\c078b2`, `violations: []` |
| C-079 | base run of the committed test file | `expected_tree: C:\t\c079base`, `violations: []` |

**Every one carries a committed verdict with `violations: []`.** No G9 proof in this correction wave
rests on an unpinned run, so none is retroactively in doubt. The sprint's own mitigation held while
the document that describes it had gone stale.

### What needs deciding, and by whom

One of two things is true and the doc owner must say which:

1. **The prohibition is superseded.** The resolved-path pin makes the `pythonpath` ordering hazard
   moot, C-070's collection fix is worth keeping, and rule 10's two sentences should be rewritten to
   say so — stating that `pythonpath = src` is present, why it is safe, and that the pin is the
   control.
2. **The prohibition still stands.** Then `pytest.ini:8` must be removed and C-070's collection
   defect solved another way, and every card merged since `5bc600e` whose G9 proof ran unpinned
   would need re-examination.

**Option 1 is the reading the evidence supports**, but the decision is the doc owner's, not the
orchestrator's — rule 10 is a merge gate and rewriting a gate to match the code is exactly the move
the sprint forbids doing unilaterally.

### Why it does not block T-106

T-106 is a benchmark run, not a G9 proof. It executes no base-vs-tip comparison and depends on
neither the prohibition nor `pythonpath`. The correction-wave proofs it rests on are all pinned and
audited above.

**Standing instruction until this is ruled on:** every base-tree measurement continues to run
through `pinned_pytest.py` with `--expect-tree` and a committed `--pin-verdict`. An unpinned base
run is not evidence, regardless of which way rule 10 is eventually resolved.

---

## F-110 — the name gate cannot relate a trivial compound name to its formula or ion notation, and C-078 makes that bite

- **Severity** **HIGH** · **Class `product_contract_violation`**
- **Registered 2026-08-23 by the Lead Orchestrator**, surfaced by the C-078 implementer (§7e) and
  **independently confirmed by the C-078 reviewer**, which found the second instance.
- **Does NOT block T-106 — but its effect MUST be predicted before the run, not discovered after.**
- Owner: `src/t2pw/mapping/`. **Out of C-078's bounds; C-078 is what makes it visible.**

### The defect

`map_ids._enforce_shipped_identity_names` refuses an identifier when the shipped name and the
candidate names share no meaningful token. Measured on committed artifacts, it cannot relate a
trivial chemical name to a formula or an ion notation:

| leg | `entity_name` | `compared_names` | verdict |
|---|---|---|---|
| PMC12096016/research (T-105) | `ferric iron` | `["Fe3+", "Fe3+", "CPD-10134"]` | `no_shared_meaningful_token` → `kegg:C14819` refused |
| PMC13231680 ×2 (T-105) | `Zn2+` | `["Zinc (II) ion", "Zinc", "ZN%2b2"]` | `no_shared_meaningful_token` → `kegg:C00038` refused |

Both refusals are **biologically wrong**. Ferric iron *is* Fe³⁺ and *is* KEGG C14819 / ChEBI 29034 /
HMDB0012943; Zn²⁺ *is* zinc(II) ion. This is **one defect class**, not two incidents: the gate has no
way to relate a name to a formula.

### Why it was invisible until now

Before C-078, the pre-freeze name-keyed re-resolution silently **reversed** these refusals — so the
row shipped correct identifiers and the bad refusal never surfaced. That masking is exactly what
C-078 closed. The mask was not benign: the same seam also reversed *correct* refusals, and it did so
onto a **different record** than the one refused (the DHB rows refused `pathbank_compound_id: 40770`
while the resolver offered `41128`).

So F-110 is a pre-existing defect that C-078 converts from *hidden and self-cancelling* into *visible
and honest*. That is the right direction — `PRODUCT_CONTRACT` §2 ranks correctness above depth — but
the visible cost lands on T-106.

### THE T-106 PREDICTION — record this before the run

`Fe3+` escaped nothing: C-078's guard fires on it, once, on the committed corpus. **`Zn2+` escaped
C-078 only because its DB match came back `ambiguous`.** On a T-106 draw where the same match
resolves uniquely, the guard strips it too.

**Predicted T-106 exposure: metal ions and formula-named compounds lose their PathBank identity.**
Concretely, expect on affected legs:

* a **coverage / depth** reduction on compound rows named as ions or formulae;
* affected rows carrying `db_status: identity_refused_review_required` with their extracted name
  intact — **not** dropped (merge rule 7 holds; verified behaviourally at a production entry point,
  `export ok=True` with the compound present in the frozen IR);
* **no** movement in acceptance **priority 1**, which counts *false* real identifiers. Removing a
  correct identifier removes nothing from that numerator. `Fe3+` appears nowhere in
  `t105_acceptance_report.json` — zero hits for `Fe3`, `ferric` or `C14819`.

**Classify any such drop deliberately as F-110, and never quote it as an unexplained coverage
regression or as evidence against C-078.**

### Fix, and why not before T-106

The correct fix is in `mapping/` — teach the name gate formula/ion equivalence, or admit a candidate
on corroborating external identifiers when the name comparison is uninformative. `mapping/` is a
large shared surface every leg traverses, and changing it immediately before a 20-leg release
candidate is the kind of late, broad change this sprint has repeatedly paid for. **Card it after
T-106**, with the T-106 legs themselves as its measurement.

---

## F-111 — `g11_evidence.py check` cannot see a double-allocated report slot

- **Severity** MEDIUM · **Class `process_tooling_gap`** · **Does not block T-106.**
- **Registered 2026-08-23**, surfaced by the C-078 reviewer while auditing a disclosed deviation.

During C-078, G11 slot **14** was genuinely double-allocated: `14-mono.json` (`chunkd-mono`) and a
hand-formed `14-chunkd-outer.json`, both stamped `started_at 2026-08-23T23:04:50`. The implementer
disclosed it and repaired it through the allocator (`next` → 46, then `os.replace`), and the
surviving artifact `46-chunkd-outer.json` still carries the discrepancy in its own body:
`json_report_path` reads `…/C-078/14-chunkd-outer.json`.

**Ruled non-material for C-078**: nothing was fabricated, nothing lost or overwritten
(`g11_evidence.py:216` takes `max(used)+1`, which tolerates a duplicate index), the repair used the
allocator, and the artifact self-documents.

**The finding is the tooling gap, not the incident.** `g11_evidence.py check` passed all 52 C-078
artifacts as compliant *including this one*, because **it never cross-checks a report's filename
against the `json_report_path` recorded inside it.** This class of defect is therefore invisible to
the gate and was known only because an agent volunteered it. F-071's atomic staging prevents a
*killed* job from leaving a partial report; it does not prevent a *hand-formed* path from bypassing
allocation entirely.

**Fix:** add a filename-vs-`json_report_path` consistency check to `g11_evidence.py check`, and make
a duplicate sequence index a reported non-compliance rather than a silent tolerance. Card after
T-106.

**Standing instruction meanwhile:** always pre-allocate `--json` with `g11_evidence.py next`. Never
hand-form a report path and promote it afterwards.

---

## F-112 — two committed tests are red because `runs_verify/**` grew, and both will mask a real regression

- **Severity** MEDIUM · **Class `test_accounting_staleness`** · **Does not block T-106.**
- **Registered 2026-08-23.** One instance disclosed by the C-078 implementer; the **second found by
  the C-078 reviewer**, which noted the sprint record was calling it one red when it is two.

| test | assertion | measured |
|---|---|---|
| `tests/test_compound_resolution_extraction.py::test_the_golden_covers_every_committed_leg_fixture` | GOLDEN/EXCLUDED accounting covers every committed leg fixture | fails |
| `tests/test_c030_canonical_identity_fallback.py::test_the_census_reproduces_over_the_committed_corpus` | `len(_corpus()) == 35` | `assert 70 == 35` |

**Both fail identically at base and at tip** across C-076, C-078 and C-079 measurements, so neither
is collateral from any card in this wave. Both have the same root cause: run directories committed
during the sprint (`runs_verify/2026-08-21_2239`, `runs_verify/2026-08-22_2147`) are not in the
accounting sets these tests pin.

Neither is in SMOKE, so neither blocks a merge gate. But each is a **pinned corpus census that can no
longer detect what it exists to detect** — a genuine regression in either census would now be
indistinguishable from the existing red. Re-baseline both, with the delta stated, after T-106 commits
its own run directory (which will otherwise break them a third time).

---

## F-113 — the 2026-08-23 identity ruling has no entry in `DECISIONS.md`

- **Severity** LOW · **Class `control_plane_gap`** · **Does not block T-106.**
- **Registered 2026-08-23**, surfaced by the C-080 reviewer.

The product-owner identity ruling of 2026-08-23 — that a UniProt accession may be shared by proven
aliases of one protein and by holo/apo states of one polypeptide, and that such rows must not be
flagged as accession conflicts unless biologically unrelated or cross-kind — is quoted verbatim in
`prompts/C-076.md` §1 and referenced throughout, but has **no append-only `D-xxx` entry** in
`DECISIONS.md`. It lives only in card prose, the LEDGER's C-076 row, and `FINDINGS.md`.

**Two merged cards now rest on it** — C-076 (`3b7a7b1`) and C-080 (`89aaced`) — and a third, C-073,
was corrected against the D-035 clause it interprets. A locked ruling that governs merged production
code belongs in the locked-decisions file, where the sprint's own rules say rulings live and where a
later reader will look for it.

**Product owner's to fix**; the orchestrator does not author `DECISIONS.md` entries. Recorded so the
gap is visible rather than assumed.

### A related gap in the same area

The ruling's own wording is *"biologically unrelated **or** cross-kind"*. Both seams implement
**cross-kind only**, because neither has a biological-relatedness oracle. Two genuinely unrelated
same-kind proteins fused onto one accession by a mapper bug are now invisible to the scorer **and**
to the production gate.

That mirrors the pipeline's pre-existing blind spot and is what the C-076 charter directs, so it is
not a deviation — but the "unrelated within one kind" half is **unmeasured corpus-wide**, and is
recorded here as a known gap rather than an assumed non-issue.

---

## F-114 — a `--basetemp` whose PARENT does not exist errors the run, a second way to fake a regression

- **Severity** LOW · **Class `process_tooling_gap`** · **Does not block T-106.**
- **Registered 2026-08-23**, hit and disclosed by the C-080 reviewer.

`TEST_MATRIX.md` § 0 documents that omitting `--basetemp` errors 83 tests with `PermissionError`.
There is a second failure mode with the same consequence: a `--basetemp` whose **parent directory**
does not exist errors the run outright. The reviewer's first affected-set run errored **55 tests**
with `FileNotFoundError: 'C:\t\bt\rev080e'`; re-running after creating the parent gave `339 passed`.

Both modes produce a large, plausible-looking failure count that is **infrastructure, not a test
result**, and either could be reported as a false regression by an agent that does not recognise it.
Worth one line in `TEST_MATRIX.md` § 0 beside the existing `PermissionError` note.

---

## F-115 — a fail-closed species-rename guard ends the leg as a CRASH instead of preserving the payload

- **Severity** MEDIUM · **Class `product_contract_violation`**
- **Registered 2026-08-24 from the T-106 run**, `runs_verify/2026-08-24_1428`, integration `efca465`.
- **Not attributable to the C-076…C-080 correction wave** — see below.

### The measurement

`PMC12444477/research` ended `status: error`, `failure_kind: crash`, at `post_pipeline`:

```
Post-pipeline conversion failed: AMBIGUOUS_RENAME_TARGET: renaming
'Escherichia coli K-12' to 'Escherichia coli' would merge it into
['Escherichia coli'], which another species row [already occupies]
```

The guard itself is **correct and pre-existing**: `pwml/prefreeze_resolution.py:762` and `:1112`,
introduced by **C-050h** (`999209e`, *"duplicate canonical rows refuse by name, not by diff string"*).
Refusing to silently merge two distinct species rows is exactly right — merging
*E. coli K-12* into *E. coli* would fuse two organisms in the exported graph.

### Why it is a finding anyway

**The refusal terminates the leg as a crash.** `status: error`, `failure_kind: crash`, no
`release_status`, no payload preserved as `review_required`.

That is the shape **merge rule 7** exists to prevent: *"preserves incomplete-but-correct pathways as
`review_required` rather than dropping them."* The pathway here is incomplete-but-correct in the
relevant sense — Stage 1 and the audit succeeded, and the only problem is that two species rows
cannot be canonicalised into one. A correct fail-closed guard should **keep its own name**, flag the
ambiguity for review, and let the run finish; `prefreeze_resolution.py:427`'s own comment says
`AMBIGUOUS_RENAME_TARGET` *"must keep its own name where it"* — which reads as the intended
behaviour, not termination.

Compare the sibling leg: `PMC12444477/strict` **passed** on the same paper in the same run and
emitted a 64,359-byte `pathway.review_required.pwml`. The research leg lost everything to a guard
the strict leg either did not hit or handled.

### Not caused by this wave

Nothing in C-076, C-077, C-078, C-079 or C-080 touches species canonicalisation. C-078 is the only
one inside `pwml/`, and it changes `_admit_db_identity` in `compound_resolution.py` — the
**compounds** canonicaliser. `PREFREEZE_CANONICALIZERS` carries `compounds` and `species` as separate
entries and the species path is untouched.

The guard appears in **neither** `runs_verify/2026-08-21_2239` (T-104) nor
`runs_verify/2026-08-22_2147` (T-105): `grep -rl AMBIGUOUS_RENAME_TARGET` returns nothing in either.
A T-106 draw that emitted both `Escherichia coli K-12` and `Escherichia coli` as species rows
exposed it for the first time.

### It displaced an expected outcome

`T106_PREDICTION.md` §4 predicted **2 × PMC12444477 TIMEOUT** as an expected non-finding, on F-092's
T-104 and T-105 evidence. **Neither timeout reproduced.** The strict leg passed outright and the
research leg ended on this crash instead. So F-092's surviving defect 3 was **not observed on this
run** and remains open on the earlier evidence rather than being re-confirmed here.

### What a fix would need

Keep the refusal, change the disposition — the same shape as C-077. The ambiguous row keeps its own
name, the ambiguity is recorded as a review flag, and the run completes with a `review_required`
classification. Ownership would be `pwml/prefreeze_resolution.py` around `:762`/`:1112` plus whatever
raises it into `post_pipeline`. **Card after the T-106 triage**; it is one leg of twenty and does not
change any acceptance priority.

---

## F-142 — the Glutathione strict-failure red is a STALE EXPECTATION, not a production defect

**Classified 2026-08-28** by the Lead Orchestrator, read-only, offline, at integration tip
`b7f1bea`. Evidence: `evidence/g11/ORCH-713/01-glut-classify.json`,
`02-glut-release-seam.json`, `03-glut-release-fields.json`, and the committed probes
`evidence/orch713_glut_probe.py` / `_probe3.py` with their logs.

**This closes the diagnosis half of the red registered under F-049 § 2** (*"fails at base …
Unowned. Do not let a later card discover this and mis-attribute it to its own diff"*) and named in
D-047 § 3's accepted pre-existing-red set. That record established *that* it failed and that no card
caused it. **It never established why.** This does.

### The two failing tests, and the single root cause

```
tests/test_strict_failure_replay.py::test_every_stored_strict_failure_replays_to_its_recorded_verdict[only_unrelated_reactions_survive]
tests/test_strict_failure_replay.py::test_recovered_cases_are_smaller_and_refused_cases_are_not_claimed[only_unrelated_reactions_survive]
```

Both fail on one fact: the fixture records `expect.recovers: false`, and
`quarantine_and_close(...).ok` is now `True`. The first asserts `result.ok is expect["recovers"]`;
the second asserts `(result.ok and shrank) is expect["smaller"]`, and `shrank` is true because two
of three reactions are quarantined. **One cause, two symptoms.** The other seven cases in the
fixture and the other three tests over this case all pass.

### The earliest failing seam — and it is not a defect

**The coverage gate fires correctly and completely.** Measured:

```
coverage.core_accepted_processes      = 0
coverage.auxiliary_accepted_processes = 1
coverage.minimum_core_satisfied       = false
coverage.reasons = ["core_process_count_below_minimum:0<1",
                    "requested_core_coverage_below_minimum:0.000<0.500"]
```

`evaluate_core_coverage` is right about everything. The two Glutathione reactions are quarantined
`quarantined_unmapped_entity` / `undeclared_entity_in_inputs`; the off-topic `citrate isomerisation`
survives as `auxiliary_accepted`; zero core processes survive; all three requested-core anchors are
unmatched.

**What changed is where that verdict goes.** `C-041a` (`4177fe5`, under **D-002**) deliberately
split the six refusal reasons at this one seam. The five that say the graph is *wrong or
unserializable* still refuse. The sixth — `minimum_core:*` — moves to a new `review_reasons` list
when the graph has a defensible core:

```python
verdict = coverage_verdict(coverage)
defensible_core = bool(verdict is not None and verdict.has_surviving_core)
review_reasons  = coverage_reasons if defensible_core else []
refusal_reasons = ([] if defensible_core else list(coverage_reasons)) + structural_reasons
```

`ok` is `not refusal_reasons`. **`ok` no longer answers the question the fixture asks it.**

### The protection did not vanish — it moved, and it is richer — ⚠ CORRECTED BELOW

At the release seam, measured on this exact payload:

```
release.status                    = "review_required"
release.strict_acceptance_eligible = false
release.completeness               = 0.0
release.missing_anchors            = ["L-glutamate","glutathione","glutamate-cysteine ligase"]
release.expansion_blocked_reason   = "3 requested-core anchor(s) matched no admitted process: …;
                                      candidate processes withheld by strict admission
                                      (quarantined_unmapped_entity:2); admitting them would
                                      require unsupported biology"
review_reasons  = ["minimum_core:core_process_count_below_minimum:0<1",
                   "minimum_core:requested_core_coverage_below_minimum:0.000<0.500"]
refusal_reasons = []
```

Nothing exports this as an accepted pathway. `strict_acceptance_eligible` is **false**, and the
M-8 invariant `strict_acceptance_eligible == (status == release_ready)` holds.

**The sibling control still hard-refuses.** `every_reaction_unresolvable` — same paper, empty graph
— gives `ok=False`, `status="diagnostic_only"`, `refusal_reasons` carrying
`minimum_core:no_surviving_process`. The empty-vs-shortfall distinction C-041a drew is live and
working, which is what makes this a *relabelling* rather than a *hole*.

### ⚠ CORRECTION — C-041a moved the CHANNEL, not the STATUS

**The section immediately above is mine (Lead Orchestrator, original F-142 text), and its second
half is wrong. It is left standing rather than rewritten, because it was honest when written and
because the progression is more useful to the next reader than a clean sentence: the finding said
ONE rule, C-103 measured TWO, and REV-103 measured THREE.**

**What stands.** C-041a genuinely moved the **channel**. `review_reasons` versus `refusal_reasons`
is C-041a's split and nothing else's, `ok` is `not refusal_reasons`, and that is the whole and
correct explanation for why `ok` changed and why the two replay tests fail. Nothing in this
correction touches that.

**What is wrong.** *"The protection did not vanish — it moved"* implies the `review_required`
**status** is C-041a's doing. It is not. The status is held by **at least three independent,
individually-sufficient caps in `classify_release_status`, none of which is necessary**:

| # | cap | source | fires here because |
|---|---|---|---|
| 1 | C-041a's `not verdict.minimum_core_satisfied` branch | D-002 / `4177fe5` | zero core-accepted processes |
| 2 | the INCOMPLETE-CORE cap | F-094 / C-072, `release_status.py:1057` | declared core, three unmatched anchors |
| 3 | the CONNECTED-PATHWAY FLOOR | C-074 arm A / F-101, `release_status.py:1093` | `connected_core_below_minimum:1<2` |

The source comment at `release_status.py:1155` names **a fourth cap of the same shape** (THE
UNSTATED REQUEST, C-074 arm B / F-100), and a fifth at `:1177`; `classify_release_status` is
explicitly a stack, and it says so in its own comments.

**The controls that settle it.** C-103 measured the first (`B1`); REV-103 measured the other two:

```
B1  collapse C-041a's minimum_core_satisfied branch  -> GREEN, 40 passed
X1  disable the F-094 incomplete-core cap ONLY       -> GREEN, 40 passed
X2  BOTH disabled at once                            -> GREEN, 40 passed
    release.status  under X2 : "review_required"      <- still
    release.reasons under X2 : [..., "connected_core_below_minimum:1<2"]   <- the THIRD cap
```

`X2` green is the decisive one: **neither named rule is necessary**, and the status was measured
under X2 rather than reasoned about.

**Why the difference is a safety property and not pedantry.** A reader told *"two independent
rules"* concludes that touching either one is safe, because the other holds. The truth is that
touching **both** is also safe, because a third holds. Those are materially different licences to
change code, and only the second is true here.

**There is no coverage gap, and this is by design.** Each cap is pinned by its own suite —
`tests/test_c072_incomplete_core_demotion.py` and `tests/test_c074_strict_core_floor.py` — so the
redundancy is deliberately invisible to the replay fixture, which measures the **composite verdict
at the seam** and demonstrably fires when that seam moves (C-103 mutations A, B2, C, D). Making the
replay suite detect the individual loss of each upstream cap would turn a replay-fixture card into
a unit test for `classify_release_status`, which is not its scope.

*Measured by C-103 (`B1`) and REV-103 (`X1`, `X2`). The original overstatement is the Lead
Orchestrator's, not C-103's.*

### Why the current behaviour is authoritative and the expectation is not

1. **`CLAUDE.md` permanent merge rule 7** requires it: *"It preserves incomplete-but-correct
   pathways as `review_required` rather than dropping them."*
2. **`has_surviving_core`'s docstring anticipates this exact payload, verbatim:** *"A fragment that
   is merely shallow, or merely **not the pathway that was requested**, is still a fragment; it
   becomes `review_required`, never `diagnostic_only`."*
3. **D-002 / C-041a is a locked, reviewed, deliberate ruling** with a recorded G9 behavioural proof.

The fixture predates it and was never updated. It measures a pre-D-002 seam.

### Classification

| Question | Answer |
|---|---|
| Production code defect? | **No.** The gate computes the correct verdict and the correct consumer receives it |
| Stale expectation? | **Yes.** `expect.recovers` asks a question `ok` stopped answering at `4177fe5` |
| Fixture drift? | No. The payload is fine and still exercises the intended shape |
| Acceptance-policy ambiguity? | No. Merge rule 7, D-002 and the `has_surviving_core` docstring all agree |
| Can it affect T-107? | **No.** Offline replay fixture; scores no acceptance priority; in no chunk |
| Attributable to C-099/C-100? | **No.** Confirmed failing at `f7dc223`, before both |

### Smallest safe card — C-103, and the trap inside it

An **acceptance-instrument correction**, not a production change: re-point the
`only_unrelated_reactions_survive` expectation at the seam that now carries the verdict —
`review_reasons`, `release.status`, `strict_acceptance_eligible` — instead of `ok`.

**⚠ The obvious fix is wrong.** Flipping `expect.recovers` to `true` makes
`test_every_stored_strict_failure_replays_to_its_recorded_verdict` take its `if expect["recovers"]:`
branch, assert full strict validity, and **silently drop the `coverage_reason` assertion** — turning
the one test written to stop an off-topic survivor reading as a win into a rubber stamp. The case
exists, in its own docstring, to prevent exactly that. **The corrected test must still go red if
`strict_acceptance_eligible` ever becomes `true` for this payload**, and the card must prove that
non-vacuously by mutation.

**Sequenced after C-101 and C-102.** It is not a T-107 blocker and must not be folded into either.

### The structural reason nobody diagnosed this for weeks

`tests/test_strict_failure_replay.py` **is in no chunk** — not SMOKE, not Chunk D, not Chunk E
(`LEDGER.md` names it among the F-054 traps). It is the **third** gate-invisible file this sprint to
hide a real state change, after `test_protein_export_policy.py` twice. A red that no gate runs gets
carried as accepted noise, and "fails at base" was allowed to stand in for a diagnosis across at
least four cards. **F-049 / F-054 remain open and this is another datum for them.**

> **⚠ CLOSED by C-103** (`card/C-103-f142-replay-expectation`, from base `ad62338`).
> **Test and fixture only: zero production lines, `git diff --numstat ad62338 HEAD -- src` is empty.**
>
> **What was done.** The expectation was re-pointed at the seam that now carries the verdict.
> `recovers` keeps its meaning — `quarantine_and_close(...).ok`, "may this graph be frozen" — for all
> nine cases, and four keys were added to every case so the fixture can say what the run *is* beside
> whether it froze: `release_status`, `strict_acceptance_eligible`, `review_reasons`, `refusal_reasons`.
> `only_unrelated_reactions_survive` now records `recovers: true`, `smaller: true`,
> `release_status: "review_required"`, `strict_acceptance_eligible: false`, both `minimum_core:*`
> review reasons and an empty `refusal_reasons`. **The trap was not walked into:** the
> `coverage_reason` assertion was moved off the `recovers` branch and keyed on the fixture *declaring*
> a shortfall, so it still runs for this case, and a new unparametrized test asserts the empty-graph
> and off-topic-survivor verdicts head-on so neither can be branch-skipped.
>
> **G9 (correction of pre-existing observable behaviour).** Same file, same selection:
> **2 failed / 37 passed / 8 skipped at `ad62338`** → **0 failed / 40 passed / 8 skipped at the tip.**
> Gold-readers (22 files) **2 failed / 453 passed, exit 1** → **0 failed / 456 passed, exit 0**;
> **that is the new gold-readers baseline** and later charters must be updated. SMOKE **473 → 473**,
> unmoved: this file is in no chunk, which is the F-054 trap the finding names above.
>
> **Non-vacuity, by mutation (F-144).** Five attacks, each restored, with the tree re-verified
> clean and green afterwards. **B1 and B2 are not two attempts at one property:** B1 asks whether a
> PRODUCTION rule is load-bearing for this payload (it is not, alone), B2 asks whether a TEST
> assertion detects the verdict it claims to guard (it does):
>
> | # | mutation | result |
> |---|---|---|
> | A | `strict_acceptance_eligible=status == RELEASE_READY` → `True` | **5 failed** |
> | B1 | `classify_release_status`: the `not verdict.minimum_core_satisfied` branch → `RELEASE_READY` | **40 passed, 0 failed — see below** |
> | B2 | `review_required` → `release_ready` at the point of record | **3 failed** |
> | C | this case's `coverage_reason` → a string that cannot match | **1 failed**, exactly this case |
> | D | `CoverageVerdict.has_surviving_core` → `True` | **3 failed**, exactly `every_reaction_unresolvable` |
> | X1 | *(REV-103)* F-094 incomplete-core cap disabled alone | **0 failed** |
> | X2 | *(REV-103)* C-041a branch **and** F-094 cap disabled together | **0 failed**, status still `review_required` |
>
> **B1 corrects this finding — and C-103's own correction was itself overstated, which REV-103
> measured.** The account above attributes `only_unrelated_reactions_survive`'s `review_required` to
> C-041a's branch alone. That is true of the **channel** — `review_reasons` rather than
> `refusal_reasons` is C-041a's split and nothing else — but **not of the status**. C-103 collapsed
> that branch (`B1`) and the suite stayed green; it then concluded "two independent rules", naming
> the F-094 incomplete-core cap (C-072) as the second. REV-103 ran the two controls C-103 did not:
> `X1` disabled the F-094 cap alone (**green**) and `X2` disabled **both** (**green**, with
> `release.status` measured as `review_required` and `connected_core_below_minimum:1<2` appearing in
> `release.reasons`). **Neither named rule is necessary.** The correct statement is **at least three
> independent, individually-sufficient caps, none of them necessary** — C-041a's
> `minimum_core_satisfied` branch, the F-094 incomplete-core cap (C-072, `release_status.py:1057`),
> and the CONNECTED-PATHWAY FLOOR (C-074 arm A / F-101, `release_status.py:1093`) — with a fourth of
> the same shape named at `:1155`. The safety difference is the point: a reader told "two rules"
> concludes touching **either** is safe because the other holds; in fact touching **both** is safe,
> because a third holds. **No coverage gap results:** each cap is pinned by its own suite
> (`test_c072_incomplete_core_demotion.py`, `test_c074_strict_core_floor.py`), so the redundancy is
> invisible to this replay fixture **by design** — it measures the composite verdict at the seam, and
> mutations A, B2, C and D show it fires when that seam moves. See the CORRECTION section in F-142's
> body above. B1's report is kept beside B2's rather than replaced, which is why any of this was
> findable.
>
> Evidence: `evidence/g11/C-103/01`–`14`, pins in `evidence/g11/pin/C-103/`, probes and logs at
> `evidence/c103_db_state.*`, `evidence/c103_replay_seam_probe.*`, `evidence/c103_gates.log`.
> Worktree had **no `.env`, no `.venv`, no `PATHBANK_DB_*`**; `resolution_db_configured()` is `False`,
> so every number here is offline. `evidence/g11/C-103/10` recorded an **invalidated** run — a
> `git checkout` restoring mutation C reverted the fixture edit with it — and is kept beside its
> re-measurement at `11`.

---

## F-144 — a non-vacuity guard can be real and still guard the wrong emptiness

**Registered 2026-08-28** by REV-101 during C-101 correction rounds 2 and 3. **The fourth and most
refined instance of this family on this sprint, and the first one caught by mutation rather than by
reading.**

### The instance

C-101 needed to pin a structural fact: under D-074 as ruled, `_contract_adjustment` can never fire,
so `accepted ≡ raw` for Priority 1. Two tests were written to guarantee that the fact stays pinned.
Both were vacuous.

* `test_a5_bare_means_bare…` called the production entry point, so it *looked* right. But it asserted
  `all(t == "" for t in tolerances(row))` where `tolerances(row)` was `[]`. **`all([])` is vacuously
  true.**
* `test_a5_no_row_shape_can_be_contract_adjusted_under_the_current_gold` carried an explicit
  non-vacuity guard, `assert seen`. `seen` was **1**, not 7 — and that single finding came from the
  `placeholder_claims_real_identity` branch, whose `contract_tolerance` is a **hard-coded `""`
  literal** and which **never calls `_contract_adjustment` at all.**

The mechanism behind both: `_contract_adjustment`'s only call site sits in the
`false_real_identifier` branch, which is entered only for a row whose name the gold **forbids** *and*
which carries an external id. `Unknown` is not in PMC12444477's `forbidden_identifiers`, so no
sentinel row ever arrived. **The tests scored rows that structurally could not reach the code they
were named for.**

### What made it visible — and nothing else would have

**Mutation, performed by someone other than the author.**

| Mutation | before the fix | after |
|---|---|---|
| guard reverted to the broader `_REAL_ACCESSION` form (**proven reachable** for five namespaces) | 38 passed | **2 failed** |
| **bareness guard deleted entirely** | 38 passed | **2 failed** |

The test's own docstring claimed *"whoever widens the licence later must come here and change this
assertion on purpose."* The reviewer widened it maximally and **nobody had to come here.**

### The class

**Asserting that *a* finding was produced is not evidence that *the path under test* produced it.**
A non-vacuity guard can be present, sincere, non-trivial, and still be satisfied by a different code
path than the one it exists to protect.

Where a null or negative result is load-bearing, the assertion must:

1. **run the production path**, not a reconstruction of it;
2. require a finding **of the specific kind that path emits**, asserted *before* anything is asserted
   about its content;
3. **survive a mutation of the thing it claims to detect.**

### The remedy, in the reviewer's words

> **A non-vacuity guard is not evidence until a party who did not write it has failed to defeat it.**

That is the part worth keeping. Round 2's tests were **sincere and wrong**; the only thing separating
them from round 3's was an adversarial mutation by someone other than their author. The author could
not have found this by trying harder — the test looked correct, passed, was not trivial, and had a
guard.

### Its ancestors on this sprint, now four

1. a guard demonstrated against a case that **could not exercise it** (REV-095);
2. a probe that **passed its own positive control and was still wrong** — case-sensitive `\bLpp\b`
   against a lowercase token, found a day after the control standard was written;
3. a test **named for a function it never called** (C-101 round 1);
4. **this** — a test that called the right entry point, over rows that could not reach the branch,
   with a guard satisfied by a different branch.

**The standard has graduated.** It began as *"a committed probe reporting a zero carries a positive
control, or its number is a ceiling."* It became *"a control proves the instrument can report
non-zero; it does not prove it is asking the right question."* It is now: **the control must exercise
the same predicate, on the same path, and survive an adversarial mutation by a non-author.**

### Practice adopted

Every card in this wave from C-102 onward is dispatched with requirement 1–3 above stated
explicitly, and is required to **paste its mutation proofs** — break it, confirm red, restore, verify
the tree clean — before reporting. See D-078.

---

## F-145 — `DECISION-BUNDLE-F132-PRIORITY1.md` § 2's population is an undercount on both axes

**Measured by C-102, independently replayed by REV-102.** The bundle records **62** gold-forbidden
terms across **32** legs and **6** papers. The corrected figures, on `bcf9a23`:

| | bundle § 2 | measured |
|---|---:|---:|
| gold-forbidden terms drawn as requested core | 62 | **92** |
| legs affected | 32 | **47** |
| papers affected | 6 | **7** |

**Two independent causes, both established rather than assumed.**

**1. The probe only ever counted UNMATCHED terms.** `orch702_f132_forbidden_anchors.py` iterates
`coverage["unmatched_terms"]`, so **26 forbidden terms that the pipeline actually matched were
structurally invisible to it.** `66 unmatched + 26 matched = 92`. The seventh paper, **`PMC13231680`**
(3 legs), is forbidden-matched on every one of its legs and therefore never appeared. The probe is
sound for the question it asked; the question was narrower than the record claimed.

**2. The committed log was stale at the moment it was committed — and that is my error.** The probe
replayed unchanged on `bcf9a23` gives **54 legs / 304 terms / 66 forbidden** against the log's
**52 / 281 / 62**. The artifact set grew by two legs (`runs_verify/2026-08-27_1341` for PMC12096016
and PMC12782028). But REV-102 established the sharper fact: the probe and its log were committed by
me at **`f71d686`**, at which point the tree **already carried 62 `quarantine_report.json`
artifacts**. **I recovered a log from a dead session's scratchpad and committed it without re-running
the probe against the tree I was committing it into.**

**The lesson, and it is the mirror image of the one that made the recovery worth doing.** Recovering
the probe was right — a G11 report certifies a job was clean and preserves nothing about what it
found. But **a recovered artifact records the tree it was run against, not the tree it lands in.**
Re-run a recovered probe before committing it, or label it with the SHA it was measured at. I did
neither.

**Consequence for the record:** the bundle's § 2 figures stand as *what was measured at the time by a
probe scanning unmatched terms only*. They are **not** the population. Any future work on F-132
should quote 92 / 47 / 7 and cite C-102's A/B.

---

## F-146 — C-074 arm B demotes the strict release status but does not stop research retaining the invented reaction

- **Severity** HIGH · **Class `product_contract_violation`** · **Registered 2026-08-29 from T-107 (ORCH-716)**
- **F-100's open remainder.** Same paper, same class, the leg arm B does not reach.
- Evidence: `runs_verify/2026-08-28_1816`, `PMC13231680/research`; `evidence/t107_score_priorities.log`

### The measurement

T-107's Priority 2 has exactly one counted row:

```
rank 2: zero unsupported retained reactions
    observed = 1   counted = 1   papers = ["PMC13231680"]
```

`PMC13231680/strict` retained **zero** reactions, so the counted row is the research leg's only one:

```
phthalylsulfacetamide decomposition to sulfacetamide     enzyme: NDM-1
```

**At T-105 the same reaction carried NO enzyme.** T-107 attached NDM-1 to it. NDM-1 is a
metallo-beta-lactamase; the paper's claim about it is that it hydrolyses **meropenem**. The gold
records **`supported_reactions: null`** for this case — nothing here is a supported reaction — and
`export_rationale` says *"The correct pipeline outcome is an empty pathway plus a rejection reason."*
PRODUCT_CONTRACT § 2 requires *"correct enzyme, modifier, transporter and cargo relationships"*.

### Why C-074 arm B did not catch it

Arm B (`release_status.py:1146-1174`, `request_was_never_stated`) is merged and working: it demotes
`RELEASE_READY` to `REVIEW_REQUIRED` when a context declares a core while naming no pathway. **It
acts on the release STATUS.** It does not stop a leg **retaining** the reaction, and **research mode
is fail-open by design** (`driver.py:2510`, `blocking_gate = gate_failed and not research_mode`), so
the research leg runs to completion and the retained row is scored.

**Arm B closed the `release_ready` route that F-100 measured. It did not close the retention route.**
The strict leg is now correct on this paper; the research leg is not.

### This single row is why T-107 is NOT ACCEPTED

Priority 2 is absolute. Six legs were eligible and one failed, so D-075's `CONDITIONALLY SATISFIED`
is unavailable — *"An eligible leg that fails remains a failure."*

**Do not fix this by suppressing research-mode retention wholesale.** Research mode is fail-open on
purpose and PRODUCT_CONTRACT § 12 requires research artifacts *"where possible"* even on
`partial_only` papers. The question is whether an **unsupported enzyme attribution** may be attached
to a retained reaction on a case whose gold declares no supported reactions at all — and that is a
narrower question than "should research mode retain anything".

---

## F-147 — the run is failed on a superseded `audit_round` report, and the failure is reported under the gate that passed

- **Severity** HIGH · **Class `product_contract_violation`** · **Registered 2026-08-29 from T-107 (ORCH-716)**
- **REGISTERED AND DELIBERATELY NOT CHARTERED.** See "Why no card" below — this is the load-bearing
  part of the finding.
- Evidence: `evidence/orch716_stale_verdict_probe.py` + `.log`, `evidence/g11/ORCH-716/10-stale-verdict-probe.json`

### The measurement

In **every** leg of T-107, passing and failing alike, `final_stage3_gate_report.json` reports:

```
ok: true    errors: []    phase: final_pre_export
```

The pass/fail difference comes **entirely** from `post_normalization_contract_report`, stamped
**`phase: audit_round`**. `streamlit_app.py:4055-4060` documents that stamp in terms:

> *"This report is still **not a verdict about what shipped** -- the remap below moves the payload
> again -- which is what the phase stamp says."*

`batch/driver.py::_blocking_reports` scans every `*_contract_report` and fails the run on any
carrying errors. **It never reads the phase stamp.** `_finalize_gate_failure` then renders the
message as *"N blocking issue(s) at `final_pre_export_stage3_gates`"* — naming the one report that
said `ok: true`. **That attribution is factually wrong** and it sent this triage to the wrong seam
before the artifacts were read.

### Stale, not a key mismatch — and the difference was measured, not assumed

Two readings fit: the audit-round payload no longer exists (**stale**), or the gate reads a field the
final payload does not populate (**key mismatch**, which would mean the final gate is blind and is
the worse defect). The probe applies the **real production predicates** —
`protein_external_identity`, `protein_species_context`, the same ones `process_normalizer.py:4627-4633`
calls — row by row to the shipped `final_mapped.json`:

| Leg | Run verdict | Shipped payload under the production predicates |
|---|---|---|
| `PMC12452463/strict` | **FAIL** | **PASSES**, 0 objections |
| `PMC12180156/strict` | **FAIL** | **PASSES**, 0 objections |
| `PMC12856317/strict` (control) | PASS | PASSES, 0 objections |
| `PMC12782028/strict` (control) | PASS | PASSES, 0 objections |

**Stale confirmed. No key mismatch.** Concretely:

- **`Fur`** was removed before export by `pre_export_strict_quarantine`,
  `degree_zero_after_quarantine`, `had_external_identity: false` — and **F-141 already ruled both Fur
  rows correct withholding.** The run was failed on an entity that does not ship and correctly has no
  identifier. PRODUCT_CONTRACT § 15: *"do not penalise the pipeline twice for obeying the contract."*
- **`ALAS2`** carries a **verified** `uniprot: P22557` / `pathbank_protein_id: 17` /
  `verification_status: verified` in the shipped payload. **The identifier was never lost** — it was
  not yet resolved when the audit snapshot was taken. Note `uniprot_id` is `None` on that row while
  `uniprot` is set: **a reader checking only `uniprot_id` concludes the identifier is missing.** That
  is F-144's trap, and the probe prints the populated key for every row so it cannot recur.

PRODUCT_CONTRACT § 1 names both halves of this among the outcomes that may never end a run without a
PWML: *"a **missing or stale gate report**"* and *"an **irrelevant degree-zero entity**"*.

### Why no card — merge rule 6, and it is not close

F-147 fails exactly two legs. **If the driver stopped honouring superseded `audit_round` reports,
both would PASS, and both would export content their own gold forbids:**

- `PMC12452463/strict`: `enterobactin synthase complex` (a `forbidden_identifier`, *"A complex name
  explicitly denoting three proteins"*); `RyhB inhibits EntC`/`EntF` (`forbidden_identifier`, *"A
  small RNA, not a protein and never an enzyme"*); an `Enterobactin secretion` transport where the
  gold notes say *"Export of enterobactin from the cytoplasm is never described at all, so no efflux
  step may be emitted"*; and an `Unknown`-backed protein where `unknown_backed_proteins_acceptable:
  false`. **PRODUCT_CONTRACT § 13 also rules this paper "Never strict success."**
- `PMC12180156/strict`: the `ferrochelatase reaction` built on **`protoporphyrin IX`**, the gold's own
  *"HALLUCINATION TEST: zero occurrences in the entire 67,304-character file"*.

**The earliest unsafe seam is Stage-1 extraction on both papers, not the driver.** Fixing the
reporting seam first would convert two contract-correct no-export outcomes into two contaminated
exports — *"weakening a biological gate to increase PWML production"* by another route, and
*"repairing downstream serialization when the earliest unsafe seam is upstream"*.

**F-147 may be fixed only after that upstream content is stopped, and the fix must land together
with the gates that would then block these legs on their real problems.** Recording it with the
probe that proves it is the deliverable; fixing it now is the mistake.

---

## F-148 — a timed-out leg preserves the stop reason and nothing else

- **Severity** MEDIUM · **Class `product_contract_violation`** · **Registered 2026-08-29 from T-107 (ORCH-716)**
- **Closes F-092 defect 3 by the same measurement.**
- Evidence: `runs_verify/2026-08-28_1816/manifest.jsonl`, the three `status: timeout` rows

### F-092 defect 3 is CLOSED

F-092 defect 3 — *"the inner deadline path records no terminal reason at all"* — was a
`product_contract_violation` authorized for code after T-106. At T-105 the inner row carried no
`termination_reason`, no `operational_failure`, no `budget`. At T-107:

```
PMC12444477/strict   stage=input   termination_reason=operation_timeout   operational_failure=true
   budget_unrecorded: "not recorded on the in-process timeout path: this seam is handed the timeout
   detail only, never the leg budget ... so elapsed and remaining cannot be stated truthfully here
   and are not guessed."
```

That is the contract's `operation_timeout` used correctly for the first time in this sprint, with
the missing budget **declared rather than fabricated**. Defects 1 and 2 stand as previously ruled
(`policy_disagreement`, no code).

### What remains

All three timed-out rows — `PMC12444477/strict`, `PMC12444477/research`, `PMC12096016/strict` —
carry `files: []` and `counts: {}`, and the outer message says *"produced nothing"*.
PRODUCT_CONTRACT § 1 names *"a **timeout without usable checkpoints or recovery information**"* an
unacceptable terminal blocker, and § 9 requires preservation of **seven** things on timeout or budget
exhaustion. T-107 preserves **two**: the exact stop reason, and the budget on the two outer rows. It
preserves **no payload, no retrieved evidence, no attempt/prompt/model/response-hash record, and no
skipped-next-step record**; `stage` is `unknown` on two of three.

**This is why `LpxH` is unverified on T-107** — both `PMC12444477` legs timed out with no payload to
inspect. It remains verified at the merged tip on the pinned run `runs/2026-08-02_2130`. **T-107 must
not be reported as confirming it.**

### A second, smaller gap in the same rows

All three carry `leg_timeout_overridden: true`, `leg_timeout_seconds: 1800.0` against
`leg_timeout_default_seconds: 3600.0`, with **`leg_timeout_override_reason: ""` and
`leg_timeout_override_source: ""`**. § 9 requires overrides to be *"explicit and recorded in the run
manifest"*. The fact and the value are recorded; the justification and provenance are empty strings.
The override **shortens** rather than extends, so *"no silent extension of difficult benchmark legs"*
is not violated — but half the requirement is unmet.

---

## F-149 — both cap tests pin non-vacuously; F-142's no-coverage-gap conclusion stands

- **Severity** n/a · **Class: audit result, no defect** · **Registered 2026-08-29 (ORCH-716)**
- Auditor: the Lead Orchestrator, who wrote **neither** file — which is what F-144 requires.
- Evidence: `evidence/orch716_nonvacuity_predictions.md` (written first, unedited),
  `evidence/orch716_nonvacuity_results.md`, `evidence/g11/ORCH-716/03..07`

REV-103 did not audit whether `test_c074_strict_core_floor.py` and
`test_c072_incomplete_core_demotion.py` pin their caps non-vacuously. **They do.**

| Mutation | Result |
|---|---|
| baseline | 42 passed |
| M1 — `MIN_CONNECTED_CORE_REACTIONS` 2 -> 1 | 14 failed |
| **M2 — arm A application forced false, constant LEFT AT 2** | **13 failed** |
| **M3 — C-072 application forced false** | **5 failed** (4 in its own file) |
| restore | 42 passed |

**M2 is the finding.** It separates *pinning a constant* from *pinning a behaviour*: anything
asserting only `MIN_CONNECTED_CORE_REACTIONS == 2` still passed, and 13 tests went red anyway —
including the file's own four `test_nonvacuity_c092_*` guards, which went red rather than staying
green. Both files exercise the **production demotion path**. `test_c072`'s
`test_the_committed_t104_leg_replays_to_the_contract_outcome` replays a **committed real artifact**,
not a fixture.

`release_status.py` was restored from **saved bytes** after every mutation (D-084 — never
`git checkout --`, never a text-mode write) and re-verified against
`sha256:db93e6f4fe30632d33725764aba668d31bfa5431f224550626f04888f0bac32d` each time. No production
line changed.

**One kept failed measurement.** The first M2 attempt used `--label nv-m2-armA-neutered`; the
uppercase `A` violates `^[a-z0-9][a-z0-9._-]*$`, the allocation was rejected, and the shell captured
an **empty string** that reached `bounded_run.py` as `--json ""`. The job ran clean and gave the
identical result but produced **no G11 artifact**, so it is uncertifiable and is not counted. It is
recorded rather than deleted. **This is the charter's named trap in a new variant: an invalid label
becoming an empty `--json` path rather than error text.** On a case-insensitive filesystem the
re-run's pin verdict also landed on the first attempt's filename,
`pin/ORCH-716/05-nv-m2-armA-neutered.pin.json` — one file, holding the valid run's verdict,
carrying the failed attempt's spelling.

---

## F-150 — two gold gaps let the run's worst false accession and its fabricated heme chemistry both score zero

- **Severity** MEDIUM · **Class `gold_data_defect`** · **Registered 2026-08-29 from T-107 (ORCH-716)**
- **PREPARED, NOT APPLIED.** A gold edit requires the product owner's authority. The exact edit and
  its A/B plan are below so the decision can be taken on a concrete proposal.
- Found by the independent `pwml-bio-auditor` adjudication; **every claim below re-verified by the
  Lead against the artifacts and the live `goldset` API before registration.**

### Gap 1 — a missing spelling means the worst false accession in the run counts as zero

`runs_verify/2026-08-28_1816/papers/PMC12180156/research/final_mapped.json`,
`/entities/compounds/2`:

```json
"name": "δ-aminolevulinic acid",
"mapped_ids": {"hmdb": "HMDB0001149", "kegg": "C00430", "chebi": "17549", "pubchem": "137",
               "cas": "106-60-5", "biocyc": "5-AMINO-LEVULINATE", "chemspider": "134",
               "drugbank": "DB00855", "pathbank_compound_id": "894"}
```

**Nine identifiers on a metabolite that occurs zero times in the source paper** — five of them
(`hmdb`, `kegg`, `chebi`, `pubchem`, `drugbank`) inside the scorer's own recognized accession set.

Measured live against `t2pw.bench.goldset.load_gold_set()`:

```
forbidden_match('5-aminolevulinic acid')     -> '5-aminolevulinic acid'  kind='placeholder_product'
forbidden_match('ALA')                       -> '5-aminolevulinic acid'  kind='placeholder_product'
forbidden_match('protoporphyrin IX')         -> '5-aminolevulinic acid'  kind='placeholder_product'
forbidden_match('δ-aminolevulinic acid')     -> None
forbidden_match('delta-aminolevulinic acid') -> None
```

Priority 1 increments `false_real` only for a **forbidden-matched** row carrying external ids, so
this row was never counted. T-107's Priority 1 = 5 with `papers = ["PMC12782028", "PMC12856317"]`;
`PMC12180156` contributes nothing.

**This is a gold-list gap, not a scorer bug.** PRODUCT_CONTRACT § 7 as ratified under D-080 says
forbidden terms are matched *"by name or declared alias and **never by resemblance**"*. The scorer
is obeying that exactly. The gold author already used the delta spelling elsewhere **in the same
case** — `acceptable_enzymes[1].aliases` carries *"erythroid delta-aminolevulinic acid synthase"* —
which makes the omission an oversight rather than a policy choice.

### Gap 2 — a bare ceiling certifies two reactions that were never extracted

`PMC12180156` sets `max_retained_reactions: 2` with **no `supported_reactions` list**. The gold
`notes` say what the 2 is for: *"the SHMT2 serine-to-glycine conversion and the SFXN1 serine
transport step."* **Neither was extracted.** The research leg retained exactly two reactions —
`glycine to heme (ferrochelatase-catalyzed)` and `ALAS2 reaction (glycine to δ-aminolevulinic
acid)`, **both fabricated heme chemistry** — and scored `2 − 2 = 0` at `completeness: 1.0`.

**The ceiling counts rows. It cannot tell right content from wrong.**

Measured across the whole gold set: **no case sets `supported_reactions_complete: true`** (all ten
are `False`), and only **two** cases set `max_retained_reactions` at all — `PMC13231680` (0) and
`PMC12180156` (2), **both negative controls.** Priority 2 is therefore evaluable only through those
two ceilings, which is why 11 of 20 legs come back `NOT EVALUATED` on D-067 precondition 3.

### The honest consequence for how Priority 2 must be quoted

Priority 2 measured **one** unsupported reaction across T-107. It could only ask the question on
**6 of 17 legs**, and the entire absolute priority rests on two negative controls, one of which has
a ceiling that cannot distinguish invented chemistry from real. **The number 1 is real; it is not a
measure of how much invented chemistry T-107 produced.** Any report quoting it must carry that
limit with it. This does not weaken § 4 of `T107-RESULT.md` — Priority 2 still `FAIL`s on a
measured, eligible leg — it bounds what the number means.

### Proposed correction — exact, and deliberately minimal

`src/t2pw/bench/gold/pinned_v1.json`, case `PMC12180156`:

1. `forbidden_identifiers[0].aliases` — add `"delta-aminolevulinic acid"` and
   `"δ-aminolevulinic acid"`. Nothing else.
2. Add an explicit two-entry `supported_reactions` list for the SHMT2 serine→glycine conversion and
   the SFXN1 serine transport step, so the ceiling of 2 is backed by signatures rather than by
   arithmetic.

**Neither change moves a threshold.** Both make an intent the gold already states in prose
machine-checkable.

### A/B plan — mandatory, because a gold edit breaks tests SMOKE never runs

1. Capture the 22-file **gold-readers** selection at the pre-edit SHA. Expected
   **`456 passed / 8 skipped / exit 0`** — the C-103 baseline.
2. Apply the edit. Re-run the same selection. **Every delta must be explainable term by term**; an
   unexplained mover blocks the edit.
3. Re-score T-107's committed artifacts against pre- and post-edit gold and record **every leg that
   moves**. **Prediction, recorded before the edit: Priority 1 rises 5 → 6** on the
   `PMC12180156/research` row. **6 is still `PASS` under D-073 (0–6)**, so the run's verdict does
   not change.
4. Report the raw number beside the corrected one, **each labelled with the gold SHA it was
   measured against**. A Priority-1 number that moved because the gold changed must never be
   reported as a pipeline regression — that inversion is the whole reason this is a prepared
   proposal and not an applied edit.

### A third item, escalated rather than proposed

`semantic.py::_external_ids` reads only `uniprot / drugbank / hmdb / kegg / chebi / pubchem`. On the
**strict** leg, `protoporphyrin IX` **does** match the forbidden list but carries only
`pathbank_compound_id: 163`, so it lands as `forbidden_identity_present_unmapped` rather than as a
Priority-1 row. **Whether a PathBank compound id is a "real external accession" for Priority 1 is a
`policy_disagreement`, not a defect.** Product owner decides; no edit is proposed here.

---

## F-151 — committing a benchmark run turned two tests red, in a file no gate runs

- **Severity** MEDIUM · **Class `product_contract_violation`** (of merge rule 4's pinned-baseline
  discipline, not of the biological contract) · **Registered 2026-08-29 (ORCH-716)**
- Surfaced by the **C-104 implementer** while establishing its own base measurement, reported
  rather than worked around. **Re-measured independently by the Lead before registration** —
  `evidence/g11/ORCH-716/13-c102-base-red.json`.

### The measurement

At the integration tip, `tests/test_c102_coverage_denominator.py`:

```
FAILED tests/test_c102_coverage_denominator.py::test_10_f132_population_regression_over_the_six_papers
FAILED tests/test_c102_coverage_denominator.py::test_13_the_accepted_rate_is_a_rate_on_every_committed_leg
E       assert 72 == 62
2 failed, 12 passed
```

**Cause, verified:** commit `e77ad3d` ("T-107 official result") committed
`runs_verify/2026-08-28_1816`, which contains **10** `quarantine_report.json` files. The tracked
population went **62 → 72**:

```
git ls-files | grep -c quarantine_report.json   ->  72
runs_verify/2026-08-28_1816                     ->  10 of them
```

Two tests pin that census with `==`. **The run commit was correct; the pins were not written to
survive it.**

### Why nobody saw it

`tests/test_c102_coverage_denominator.py` is in **neither SMOKE nor gold-readers**. It is in no
chunk. The tip has been red here since `e77ad3d` and every gate this sprint has stayed green.

### The file already contains the right idiom, three lines away

```python
line 325:  assert len(paths) >= 62, "the committed artifact population shrank; re-pin before reading on"
line 347:  assert legs == 62
line 461:  assert checked == 62
```

**Line 325 is `>=` and its message says the guard exists to catch the population *shrinking*.**
Lines 347 and 461 assert `==` for the same corpus and the same purpose — they are non-vacuity
guards making sure the loop actually checked something. **A `>=` preserves that purpose exactly and
survives a corpus that grows every time a run is committed, which is the corpus's normal
behaviour.** The inconsistency is inside one file, written by one author, for one census.

### Consequence beyond the two reds

**`evidence/c102_mutation_attack.py` cannot run at all.** It asserts `code == 0` on its unmutated
baseline before applying any mutation, so it aborts immediately
(`evidence/g11/C-104/09-attack-set-baseline.json`). **The sprint's mutation-attack driver has been
unrunnable since `e77ad3d`** — which matters because D-078 and F-144 make mutation testing a
required practice on every card. C-104's R5 entry is correct and its substitution verified
statically, but it cannot be exercised through the driver until this is fixed.

### Proposed correction — narrow, test-only, and it matches the file's own idiom

`tests/test_c102_coverage_denominator.py`: change lines 347 and 461 from `== 62` to `>= 62`,
carrying line 325's existing justification. **Do not re-pin to 72** — that only moves the breakage
to the next committed run. Do not delete the assertion; it is a real non-vacuity guard.

**This is merge rule 4's own escape hatch used correctly:** a pinned baseline moved deliberately,
with an exact documented delta (62 → 72, attributable to exactly ten legs from one named run).

### What this does NOT license

It does not license loosening any *biological* pin to `>=`. The census here is an artifact count —
"did the loop see the whole committed corpus" — not a measurement of pipeline quality. A `>=` on a
false-identifier count or a retained-reaction count would be the merge-rule-6 direction and is a
different thing entirely.

### Standing lesson

**A committed benchmark run changes tracked corpus counts, and a test that pins one with `==` will
go red on the next run — silently, if the file is in no gate.** Before committing a run's
artifacts, grep the test tree for `==` pins on committed-artifact censuses. Every such pin should be
`>=` with a stated floor, or derived.

### CORRECTION, same day — my proposed fix was wrong, and REV-104 said why

**The `>= 62` proposal above is withdrawn.** It is left standing rather than edited out, because the
reason it is wrong is the useful part.

REV-104, asked whether F-151 should fold into C-104 or stand alone, answered *separate card* and
gave a third reason I had not considered:

> *"`>= 62` is not obviously right, and that is the real reason. Line 325 uses `>= 62` to catch the
> population **shrinking**. But `test_10` and `test_13` pin **derived** quantities (`legs`,
> `checked`) **against that census**; relaxing them to `>=` means the suite stops noticing that ten
> new legs entered the population unremarked."*

That is correct and my reasoning was not. I generalised from line 325's idiom without checking what
lines 347 and 461 are *for*. Line 325 guards a **floor** — did the corpus shrink. Lines 347 and 461
assert that a loop **visited every leg it should have**, and the census is how they know how many
that is. A `>=` there would let a leg join the corpus and go unvisited without anyone noticing,
which is the opposite of the guard's purpose and is a quiet vacuity of exactly the F-144 shape.

**Revised proposal: re-pin to 72, and record why it grew.** That is what the file's own comment
("re-pin before reading on") asks for, and it keeps the pin doing its job. The cost is that the pin
must be moved deliberately every time a run is committed — **which is the correct cost**, because
moving it is the moment someone confirms the new legs were meant to be there.

**Revised disposition: its own card, and larger than the pin.** REV-104's recommendation, which I
accept:

> *"One small card, 'make `c102_mutation_attack.py` runnable and D-084-compliant.' It decides and
> fixes the census pin, replaces `git checkout --` + text write with a saved-bytes binary restore,
> and then **executes the harness with all seven mutations including R5**."*

That is what actually discharges C-104's A2 intent — the next reviewer inheriting a **runnable**
mutation rather than a registered one. **Not chartered in this wave**; handed forward with the
analysis complete.

**Measured, by REV-104, in the same file:** the harness's restore converts the whole file's line
endings, and then `git checkout --` is the only thing that puts them back —

```
on disk        : bytes=79745  crlf=1673  bare_lf=0
harness write  : bytes=78077  crlf=0     bare_lf=1673
```

**Both rows of D-084's table, in one loop**, in the instrument the sprint uses to certify that its
guards are non-vacuous.

---

## F-152 — C-104's widened abort guard can fire on a green file, because the count parse reads all of stdout

- **Severity** MEDIUM · **Class `product_contract_violation`** (of the gate's own honesty, not of the
  biological contract) · **Registered 2026-08-29 (ORCH-716), found by REV-104**
- **No live exposure today**, and the failure mode is loud rather than silent.

`evidence/c102_goldreaders_split.py:52` parses counts with
`re.findall(r"(\d+) (passed|failed|skipped|errors|error)", out)` over the **entire** combined
stdout+stderr, not over pytest's summary line. So a passing file whose output happens to contain the
text "3 errors" — a warning, a captured log line, a failure message — is recorded as `errors=3`.

**Before C-104 a spurious `errors > 0` was inert**: it only suppressed an abort. **After C-104 it is
fatal.** REV-104 measured it:

```
SCENARIO green_with_warning_text
  BASE exit=0 aborted=False files_reported=22
  TIP  exit=1 aborted=True  files_reported=1
       abort : INFRASTRUCTURE FAILURE ... exit=0 errors=3 failed=0 passed=2
```

The same applies to a genuine red whose failure text contains "*n* errors": that red would be
reported as an infrastructure failure and the gate would stop early instead of folding it into the
totals.

**Why C-104 was merged anyway.** The card ordered `errors > 0` to abort *"independent of exit code
and of `failed`"*, and the loose parse is pre-existing C-102 code at line 52 — **outside C-104's
declared boundary of "the abort guard, line 65"**. The reviewer correctly declined to overrule the
card's wording or to grow its boundary mid-review. The real 22-file selection shows `errors=0` on
every file, so nothing is exposed today.

**Fix:** scope the parse to pytest's summary line. Belongs with the F-151 card — same file family,
same "make the instruments trustworthy" job.

### One further behaviour change, recorded so it is not rediscovered as a defect

A file with **both** a genuine red **and** a setup error (`exit=1, failed=1, errors=1`) folded into
the totals at base and now **aborts** the gate (REV-104: `fail_plus_error BASE False / TIP True`).
This is **correct** under D-083's *"a setup error is never a legitimate outcome of this gate"*, but
it is a third behaviour change beyond the two the card describes. Recorded, not a defect.

---

## F-153 -- the map that exists to stop rebuilding told two agents to rebuild

- **Severity** MEDIUM · **Class `product_contract_violation`** (of the sprint's own
  do-not-rebuild discipline, not of the biological contract) · **Registered 2026-08-31 (ORCH-717)**
- **Surfaced by a peer session** doing read-only reconnaissance of the RAG subsystem for an
  unrelated question, and **re-verified by the Lead before registration**. Recorded because the
  route matters: this was found by someone reading the map in order to use it, which is the only
  way a stale map gets found.

### The measurement

`MASTER_PLAN.md:153`, in the section `CLAUDE.md` points every agent at with the instruction
**"Do not rebuild what exists"**:

> *"Genuinely missing in RAG: a stopping policy and a loop controller."*

Both exist, and both are wired:

```
src/t2pw/rag/loop_policy.py      14647 bytes
src/t2pw/rag/controller.py       16234 bytes
src/t2pw/rag/graph_delta.py      24722 bytes
streamlit_app.py:1270    from t2pw.rag.controller import ... run_rag_loop
streamlit_app.py:1426        outcome = run_rag_loop(
streamlit_app.py :: run_rag_rounds       rag_loop_record = run_rag_rounds(   <- SYMBOL, not a line (F-157)
tests/test_c055_rag_loop_wiring.py    exists
```

And `src/t2pw/rag/controller.py:11` still says, in its own module docstring:

> *"**UNWIRED**: nothing in production calls it; wiring is C-055's."*

C-055 is merged. The docstring is false at this tip.

### Why this is worth a finding rather than a typo fix

**A stale map is not symmetrical with a stale test.** A test that pins an old number goes RED and
someone looks at it -- that is F-151, and it was found in a week. A map that says a module is
missing goes on being read as true, and the cost lands as **duplicated work by an agent who
believed it**, months later, with nothing to attribute it to.

MASTER_PLAN section 2 exists for exactly one purpose, stated in `CLAUDE.md`: *"Do not rebuild what
exists."* On these two components it said the opposite. **The one section whose job is to prevent
duplicated work was, for these items, instructing it.**

### Disposition

* **`MASTER_PLAN.md:153` CORRECTED in this wave** by the Lead, who owns that document. The false
  sentence is struck through rather than deleted, with the wiring call sites named, so a reader who
  remembers the old claim sees it was retracted rather than silently rewritten.
* **`controller.py:11` NOT corrected.** It is a `src/` change and **no card in this wave owns that
  file**. C-106 is test/evidence tooling; C-107 owns `curation/apply_audit_patch.py` only. Fixing
  it opportunistically would be an out-of-boundary production edit, which is the thing the merge
  rules exist to stop -- **a one-line docstring is exactly the change that feels too small to
  charter and is therefore the one most likely to normalise boundary drift.** Handed forward.
* The **third** claim in the same paragraph -- that graph-delta validation is partial because
  `conform.py` conforms and merges without validating the delta against a policy -- was **not**
  re-verified and is left standing as written. Correcting two claims is not licence to certify a
  third that nobody measured.

### Standing lesson

**When a card wires a module, the same card must retire the docstring and the plan entry that said
it was unwired.** C-055 wired the loop controller and left two documents saying it had not. Neither
is covered by any gate, and no test can catch either: prose staleness is invisible to CI by
construction, so the only control is the card that changes the code changing the words in the same
commit.

---

## F-154 -- the pinned line addresses drifted ABOVE the pin, and a sprint agent reads the wrong table

- **Severity** MEDIUM · **Class `product_contract_violation`** (of the sprint's own citation
  discipline) · **Registered 2026-08-31 (ORCH-717)**
- **Surfaced by the C-106 implementer** while resolving a conflict in its own card, reported rather
  than worked around. **Re-verified by the Lead against the base tree before registration.**

### The measurement

The `## Test discipline` chunk-membership bullet of `.claude/agents/pwml-test-runner.md` (~~`:59`~~) tells that agent to certify chunk membership by a
**stem-exact** match against three addresses. Two of the three point at the wrong content, measured
at base `c7fb5c5`:

| Citation | What it claims | What is ACTUALLY there |
|---|---|---|
| `TEST_MATRIX.md:213-218` | the chunk membership table | the **bounded-runner function table** -- `launch_child`, `_kill_tree`, `child_env` |
| `TEST_MATRIX.md:242-252` | the SMOKE command block | the **Chunk-D-excluded / Chunk-E** paragraphs |
| `evidence/chunk_d_gate.py:63-70` | Chunk D's file list | correct |

The real locations, measured:

```
chunk table  : TEST_MATRIX.md:230   header, rows 232-237 (A B C D-core D-apptest E)
SMOKE block  : TEST_MATRIX.md:259   through :271
```

A consistent drift of roughly **+17 to +19 lines**. `FINDINGS.md:1120-1124` carries the same error
from the other side: it cites `TEST_MATRIX.md:213` for *"Chunk A, 6 files"*, and Chunk A is at
`:232`.

### Why this one is not cosmetic

**The instruction is a stem-exact match, and the range it names contains no test-file stems at all.**
An agent obeying `pwml-test-runner.md:59` literally would match `launch_child`, `_kill_tree` and
`child_env` against a list of test files and find **nothing** -- then either report every file as
unchunked, or report the match as empty and move on. Both are silent. Neither looks like an error.

The same line warns, correctly, **never to certify membership by grepping the filename**, because
`tests/test_map_ids.py` collides with `test_map_ids_name_gate`. So the agent is steered away from
the working method and toward a broken address.

### The irony worth recording, because it is the lesson

`TEST_MATRIX.md:538` states the constraint that exists to prevent exactly this:

> *"addresses are pinned by citation up to `:477`, so nothing may be inserted above that."*

and `:567-568` explains it again: *"Inserting there would shift every line between 209 and 477 and
break those citations."*

**The insertion already happened, above the chunk table, before the pin was written or in spite of
it.** The rule is correct and it is being obeyed going forward; what nobody checked is whether the
addresses were right **at the moment the pin was declared**. A pin freezes whatever it is pointing
at, including a mistake.

### What C-106 did, and did not do

C-106's card ordered `TEST_MATRIX.md` edits at the chunk table and the SMOKE block -- **both above
`:477`** -- which put the card in direct conflict with the `:477` rule. **The implementer flagged
the conflict rather than silently violating one of the two**, and resolved it by making every edit
above `:477` **line-neutral**: rewrite and extend in place, never add or remove a line, with the
full record appended at end-of-file.

Verified by the Lead: base and tip both have **477 lines above the pin**, line `:477` is
byte-identical, and the only changes at or below it are in-place (`239-242c`, `259c`, `271c`). The
chunk table is still at `:230` and the SMOKE block still at `:259` **at the tip**, so the drift did
not get worse.

**The card was wrong and the implementer was right.** That is recorded as a defect in the card, not
in the work.

### Disposition

**Registered, NOT fixed.** The wrong addresses live in `.claude/agents/pwml-test-runner.md` and in
`FINDINGS.md:1120-1124`, and **no card in this wave owns either**. Fixing an agent definition
opportunistically is the same boundary drift F-153 declined, and this one is larger: it changes how
a sprint agent behaves.

**The fix is one edit and the correct values are above** -- `:213-218` -> `:230-237`, `:242-252` ->
`:259-271` -- so whoever charters it does not need to re-derive them.

### Standing lesson

**A line-address pin is only as good as the addresses at the moment it is declared, and nothing
checks them.** Two documents and one agent definition have been citing the wrong table for long
enough that nobody noticed, because a citation that points somewhere plausible fails silently.
Prefer an anchor that cannot drift -- a heading, a unique string, a named table -- and where a line
address is unavoidable, **verify it at declaration time and re-verify whenever the file is edited
above it**.

---

## F-155 — three pre-existing routes past the actor-evidence guard, all needing one follow-on card

- **Severity** MEDIUM–HIGH · **Class `product_contract_violation`** (F-146's class: an actor role
  admitted without evidence for it) · **Registered 2026-08-31 (ORCH-717)**
- **(a) and (b) surfaced by the C-107 implementer, (c) by REV-107 while correcting its own fixture
  error. All three independently confirmed by REV-107 against base `33a99e7` and tip `9890770`.**
- **None is introduced by C-107.** All three reproduce identically at base and tip.

### (a) The transport family's bare schema noun self-licenses

`_ROLE_CUE_RES["transport"]` opens with the bare stem `transport`, and `transport` **matches inside
"transporter"**. So a rationale that argues purely from payload shape licenses the role it is
asking for:

```
op: add … /transporters/-   actor MsbA
evidence: "add MsbA as a transporter to resolve the structural inconsistency"
   ->  ACCEPTED at base 33a99e7 AND at tip 9890770
```

**This is exactly F-146, in a family the pinned safety property does not name.** The catalysis
control refuses the same sentence — *"add NDM-1 as an enzyme to resolve the structural
inconsistency"* — because C-105 deliberately excluded the bare schema nouns `enzyme`, `enzymatic`
and `activity` for this precise reason, recorded in its own comment:

> *"The bare schema nouns 'enzyme', 'enzymatic' and 'activity' are not cues either … a promoted
> rationale is written in exactly those words."*

**That reasoning was applied to catalysis and never carried across to transport.**

### (b) `[^.]` is a no-op in every cue pattern — there is no sentence bound

Several catalysis alternatives are written as `…\b[^.]{0,80}\bby\b` and read as *"within one
sentence"*. They are not. `_match_fold` replaces every run of non-alphanumerics with a single space
**before** the pattern is applied, so **no `.` ever survives into the haystack**. `[^.]` therefore
excludes nothing and the construct is a **length bound only**.

Consequence: a contra-cue one sentence away still fires, and a passive agent one sentence away was
matchable until C-107's 1b closed that particular case by a different route. The comment's stated
intent and the code's actual behaviour differ, which is the kind of gap that survives review because
the pattern *looks* like it does what the prose says.

Fixing it means changing `_match_fold`, which is shared, calibrated and outside any current card.

### (c) An actor whose own NAME contains an enzyme noun licenses with no cue in the span

The enzyme-noun rule is part of the catalysis vocabulary, and the cue is sought in a window around
the **matched name token** — so a name that *is* an enzyme noun supplies its own cue:

```
actor "LpxC hydrolase"
evidence: "LpxC hydrolase was quantified in the lysate"
   ->  ACCEPTED at base AND tip, with no role-predicating claim anywhere in the span
```

The span says the protein was **measured**, not that it catalysed anything. Any actor named
`… synthase`, `… transferase`, `… hydrolase` clears the guard on its name alone.

**How it was found is worth recording.** REV-107 used `"LpxC hydrolase"` as a multi-token battery
name, got a false mismatch, and traced it to the name supplying its own cue. **It corrected its
fixture and registered the property rather than discarding the anomaly** — the same discipline that
caught C-105's original preservation control passing only because it used a one-character protein
name.

### Why one card and not three

All three are **the same shape**: something that is not evidence about *this actor performing this
role* is nevertheless accepted as that evidence — a schema noun, a length bound mistaken for a
sentence bound, and a name mistaken for a claim. A card fixing one and not the others leaves the
class open, and each fix touches the same two functions.

### Why NOT chartered inside C-107

C-107 owns a declared calibration of six routed findings. **Widening it mid-card to chase three
newly-found routes is precisely how C-105's first round reached four production callers instead of
the one its card named** and refused 12 of 29 legitimate cases. REV-107's judgement, which the Lead
accepts:

> *"My judgement: it should not have blocked this card — it is unchanged by the diff, it is
> registered, and the catalysis control refuses. But it is the same failure mode the author then
> reproduced in the new cofactor family, and both belong on one follow-on card."*

### The one that is NOT pre-existing, and is therefore C-107's to fix

C-107's own `cofactor` family admits bare `requires | required for | requirement for | depends on |
dependent on | dependence on | in the presence of`, so *"the reaction requires a cofactor, so P is
added to resolve the structural inconsistency"* goes **refused at base → ACCEPTED at tip**. Because
`_ANY_ROLE_CUE_RE` is rebuilt from every family's vocabulary, it also widens the `"other"` fallback
for **every unmapped role**.

**That one is a blocking correction-round item on C-107, not part of F-155** — it is the same
failure mode as (a), which is exactly why (a) must be fixed before it is reproduced a third time.

### Standing lesson

**A bare schema noun is not evidence, and C-105 knew that — for one family.** The exclusion of
`enzyme` / `enzymatic` / `activity` from the catalysis cues is correct and reasoned, and it was
never generalised. When a guard is built family by family, **the reasoning behind each exclusion is
the thing to carry across, not the word list** — a later family written from the word list alone
inherits the vocabulary and loses the principle.

### ADDENDUM, 2026-08-31 — two more members of the class, found while closing C-107

Both surfaced by REV-107's final round, both measured, **neither blocking, both folded here rather
than reopening C-107.**

#### (d) C-105's OWN attenuation stems carry the identical unanchored defect

C-107 round 2 fixed exactly this bug in the six stems **it** had added, and anchored them as words.
**C-105's stems, one file-section above, were deliberately left untouched — and they have it too:**

```
REFUSED  "the repressor complex P4X catalyses the conversion of A to B"
REFUSED  "the suppressor protein P4X catalyses ..."
REFUSED  "the inhibitor protein P4X catalyses ..."
```

`repress` inside `repressor`, `suppress` inside `suppressor`, `inhibit` inside `inhibitor`. Present
at base, at every C-107 tip, and today. **The author was right not to touch them** — they are the
prior card's, and reaching into them mid-card is the scope creep that produced C-105's own round-1
failure. They belong here.

**This is the third independent instance of the same defect in one file** — `mediat` inside
"intermediate" (C-107's finding 1f), the six stems C-107 added, and now C-105's. Any card touching
this file should treat "is this stem anchored as a word on both sides?" as a checklist item rather
than a discovery.

#### (e) Three load-bearing anchors that no test covers

REV-107 mutated the anchors C-107 round 2 added and found three green:

| anchor removed | legitimate spans that flip ACCEPT → REFUSE |
|---|---|
| left anchor on `_ATTENUATION_WORD_SRC` | 4 — `photoablation`, `counterinterference`, `microablation`, `nonimpairment` |
| right anchor on `_ATTENUATION_WORD_SRC` | 2 — `silencer` / `blocker` after the actor, via F2 |
| left anchor on the six inhibition additions | 4 — the same four `-ablation` / `-interference` words |

**None is dead code and current behaviour is correct in every case** — the reviewer measured the
exposure rather than reasoning about it, and every flip is in the **over-refusal** direction, so
each anchor is protecting precisely the class C-107 round 2 exists to protect. What is missing is a
test, not a behaviour. Same class as the already-registered mutation-coverage gaps on the contra's
modifier path.

**Why this matters more than an ordinary coverage gap:** these anchors are the fix for a blocking
finding. An untested fix for a blocking finding can be removed by a future refactor with the whole
suite green — which is the exact shape of the `V9` gap C-107 round 2 was sent back to pin.

#### The class, restated with all five members

`(a)` transport bare schema noun · `(b)` `[^.]` is not a sentence bound · `(c)` an actor's own name
supplies its cue · `(d)` C-105's unanchored stems · `(e)` three untested anchors.

**(a)–(d) are all "something that is not evidence about this actor in this role is accepted, or
something that is evidence is refused, because a pattern matched inside a longer word or matched a
schema noun."** (e) is the coverage that would stop (d) recurring a fourth time.

**A follow-on card should take all five together.** Each fix touches the same two functions, and the
sprint has now watched this class recur three times in three consecutive cards on one file.

---

## F-156 — the third stale claim in `MASTER_PLAN` § 2 is also false, and the enforcement is load-bearing

- **Severity** MEDIUM · **Class `product_contract_violation`** (of the sprint's own do-not-rebuild
  discipline, like F-153) · **Registered 2026-08-31 (ORCH-717)**
- **Refuted on the code by the peer session `project14-t2pw-93`**, static reading only, pinned to
  `03138d1`. **Certified behaviourally by the Lead**, who ran the tests and then mutated the
  enforcement. **Both halves of that provenance are load-bearing and are recorded separately on
  purpose** — see § "Why the provenance is written this way".

### The claim

`MASTER_PLAN.md` § 2, the **third** claim in the paragraph F-153 corrected:

> *"Graph-delta validation being partial (`conform.py` conforms and merges but does not validate the
> delta against a policy)."*

F-153 deliberately left it standing: *"Correcting two claims is not licence to certify a third that
nobody measured."* **That was the right call, and this is the measurement.**

### The premise is TRUE and the conclusion is FALSE

**Premise, confirmed:** `src/t2pw/rag/conform.py` contains **zero** references to `graph_delta`,
`validate_graph_delta` or `DeltaVerdict`. It genuinely conforms and merges without validating.

**Conclusion, refuted.** Validation is not partial. It is implemented, wired, reached in production
and **enforced**, verified at `03138d1` by `git show` rather than a working-tree read:

```
implemented : rag/graph_delta.py:386  def validate_graph_delta(...)   11 rules in RULES
called      : rag/controller.py:297   validate_graph_delta(mark.graph, candidate, RoundRecord(...))
enforced    : rag/controller.py:306   if verdict.admissible:
                                          graph = candidate
production  : streamlit_app.py:5636   rag_loop_record = run_rag_rounds(...)  ->  run_rag_loop
```

A refused delta **does not advance the canonical graph**; the round is recorded via
`LoopOutcome.refused` and nothing is repaired.

### Why both are true at once — this is the part worth keeping

**Validation was never `conform.py`'s job.** `conform` builds the `{"additions": …}` envelope that
`pipeline.merge_additions` injects; the validator runs **one level up**, in the controller, comparing
the checkpoint graph against the post-stage candidate. `graph_delta.py`'s own header says so:

> *"It neither conforms nor merges: `t2pw.rag.conform` already builds the envelope."*

**So the map looked for the validator in the module that by design does not hold it, and read its
correct absence as a gap.** That is the same failure shape as the other two stale statements in the
same paragraph: the map describes the **pre-C-043 / pre-C-055 world**.

### Behavioural certification — because a static read is not proof under G9

The peer was explicit that it had run nothing and that its status was *"refuted on the code,
certification needs those tests actually run"*. It was right to say so. Two steps followed:

**1. The tests run and are green.** `tests/test_rag_graph_delta.py` +
`tests/test_c055_rag_loop_wiring.py` → **52 passed**. G11 `ORCH-717/19`.

**2. Green tests do not prove the enforcement is load-bearing, so it was mutated.**
`evidence/orch717_f156_mutation.py`, G11 `ORCH-717/21`. The guard was removed so a **refused** delta
advances the graph anyway — precisely the fail-open the claim feared:

```
baseline (unmutated)        : 52 passed
mutant (enforcement removed): 1 failed, 51 passed
  FAILED tests/test_c055_rag_loop_wiring.py::
         test_a_refused_delta_does_not_advance_the_graph_and_says_which_rule
restore replayed saved bytes: True
```

**MUTATION CAUGHT. The enforcement is load-bearing and covered**, by a test whose name states the
exact property. **D-084: the restore replayed SAVED BYTES**; `git checkout --` was not used, and the
mutation ran in a disposable worktree, never in the integration tree.

### The probe's first run is preserved, and it is the more instructive one

`orch717_f156_mutation.attempt1-lf-anchor-crlf-tree.log`. The probe's byte anchor was written with
`\n`; the tree checks out **CRLF**. It **refused to mutate and exited 2** rather than guess.

**That refusal is the behaviour to keep.** A probe that "helpfully" normalised the line endings would
have silently mutated the wrong bytes — and a mutation probe that mutates the wrong thing reports
`MUTATION SURVIVED` and is read as **missing coverage that is actually present**. The second version
detects the endings from the file and says which it found.

### Why the provenance is written this way

The peer asked, unprompted, that its caveat survive into the record — *"'session 93 refuted it' reads
stronger than what I actually did"* — and it was right to insist. **A static read and a mutation-proved
property are different epistemic objects and this sprint has been burned by conflating them.** The
refutation is the peer's; the certification is the Lead's; neither is the other.

It also found this the same way F-153 was found: **by reading the map in order to use it.** That is
the only way a stale map ever gets found, and it is now twice in one wave.

### Disposition

* **`MASTER_PLAN.md` § 2's third claim is FALSE and must be retracted** the way F-153's was — struck
  through, not deleted, with the call sites named.
* **NOT folded into C-109 mid-flight.** C-109 is dispatched with an explicit instruction *not* to
  certify or delete this claim, and that instruction was correct when written. **Widening a card
  mid-flight to chase a newly-found item is exactly how C-105's first round reached four production
  callers instead of the one its card named.** Preserving a now-refuted claim for one more card costs
  nothing; breaking a card's boundary costs a round.
* **Handed forward as a one-paragraph correction** on the next control-plane card, with this finding
  as its evidence. The measurement is done; nobody needs to re-derive it.

### Standing lesson

**A claim of the form "X does not do Y" is only as good as the assumption that X is where Y belongs.**
All three stale claims in that paragraph shared one root: they were written against an older
architecture, and the two that named a *module* were falsified by the wiring moving, not by the
functionality being absent. **When a card moves a responsibility between modules, the claim that the
old module lacks it becomes true and misleading in the same commit.**

---

## F-157 — a citation pinned to bytes that exist in no commit, in the file nobody may commit

- **Severity** MEDIUM · **Class `product_contract_violation`** (of the sprint's citation discipline —
  F-154's class, in its worst available form) · **Registered 2026-08-31 (ORCH-717)**
- **Surfaced by the C-109 implementer** while executing its charter, and **reported rather than
  silently corrected**. Re-verified by the Lead against the committed tree before registration.

### The measurement

F-153, the merged `MASTER_PLAN.md` § 2 correction, and **the Lead's own C-109 charter** all cite
`streamlit_app.py:5669` as the production call site for `run_rag_rounds`. Measured at `2ac8404`:

```
git show 2ac8404:src/t2pw/app/streamlit_app.py
  :5636  ->      rag_loop_record = run_rag_rounds(          <- the actual call
  :5669  ->  st.session_state["pathway_context"] = pathway_context

working copy (UNCOMMITTED) src/t2pw/app/streamlit_app.py
  :5669  ->      rag_loop_record = run_rag_rounds(          <- where the citation came from
```

`:1270` and `:1426` are correct in both. `run_rag_rounds` is defined at `:1239`.

### Why this is F-154's defect in its worst form

A drifted line address at least points at *something in the repository*. **This one points at nothing
any reader can obtain.**

`src/t2pw/app/streamlit_app.py` is on the sprint's **never-commit list** — a protected product-owner
diff of **35 insertions / 2 deletions** that is deliberately never committed. So the citation was
taken from a working tree whose bytes exist in **no commit and no branch**, and **no reader,
reviewer or future agent can reproduce it from git**. It is not stale; it is unreachable.

The `+33`-line offset between the two is exactly the protected diff.

### What makes it worth its own finding rather than an F-154 instance

F-154's mechanism is *insertion above a pin*, and its cure is a stable anchor. **This mechanism is
different: the address was never right in the artifact a reader has access to.** An anchor fixes it
too, but the lesson is separate and sharper:

> **When you cite a line in a file that is uncommitted — or that carries an uncommitted diff — you
> are citing your own working tree, not the repository.** Every measurement offered as evidence must
> be taken through `git show <sha>:<path>`, not by reading the file in place, whenever the file can
> differ from HEAD.

**And `streamlit_app.py` is the file in this repository most certain to differ from HEAD**, because
the contract requires it to.

### How it propagated, recorded honestly

The bad citation went **F-153 → the merged `MASTER_PLAN.md` correction → the Lead's C-109 charter**,
picked up unexamined at each step because the previous document was trusted. **The Lead had also
verified `:5636` independently earlier in the same wave** — while checking a peer session's
graph-delta claims, where the peer cited `:5636` and F-153 cited `:5669` — **and did not reconcile
the discrepancy.** The implementer did.

**That is the third time this wave a dispatched agent has corrected the Lead**, and the second time
the correction was to a number the Lead had the evidence to catch. *Reading the report is not
verification, and that applies to your own reports* — including to a discrepancy you have already
seen and not chased.

### Disposition

* **Corrected in `controller.py` and the `MASTER_PLAN.md` note by C-109**, which cites `:5636`,
  explains the discrepancy, and whose probe asserts **both halves** — that `:5636` resolves and that
  `:5669` does not.
* **`FINDINGS.md` § F-153's own text carried `:5669`** — **CLOSED by C-112.** It now cites the
  **symbol** `run_rag_rounds`, not the corrected number `:5636`: a line address inside a file that
  carries an uncommitted diff is unciteable by construction. Was outside C-109's boundary.
* **No change to `streamlit_app.py`.** It is protected and stays exactly as it is at 35/2.

### Standing lesson

**A protected uncommitted file is a citation hazard, not just a merge hazard.** The sprint already
knows never to *commit* `streamlit_app.py`; what it did not have written down is that you must never
*cite line numbers in it* either. Cite a **symbol** — `run_rag_rounds`, `run_rag_loop` — which is
identical in both trees and cannot drift by 33 lines.

---

## F-158 — `RESULT.txt` prints the empty blocks but not the two fields that say WHY they are empty

- **Severity** LOW–MEDIUM · **Class `product_contract_violation`** (§ 9 preservation, the reporting
  half) · **Registered 2026-08-31 (ORCH-717)**
- **Surfaced and measured by the C-110 implementer**, while taking that card's `batch/runner.py` stop
  condition. **Reported rather than fixed**, because the seam belongs to F-148 and the card was
  forbidden to build into it.

### The measurement

`src/t2pw/batch/runner.py::result_text(row, *, paper=None)` renders the per-paper-per-mode verdict
file. It prints `status`, `stage`, wall time, warnings, issue codes, **`counts` and `files`**.

It does **not** print **`termination_reason`** or **`operational_failure`** — *the two fields that
state in terms that a leg was an operational casualty rather than a scientific decline.*

### It corrects the Lead's framing, and the correction matters

The C-110 charter said `result_text` *"drops the operational/biological distinction entirely"*.
**That is wrong, and the implementer measured it rather than accepting it.** The distinction is
partly visible: a timed-out leg shows empty `counts` and `files`, so a careful reader *can* see that
nothing was produced. What the page never says is **why** — whether the emptiness was chosen or
inflicted.

**The gap is two named fields, not a whole missing capability**, and scoping it correctly is what
makes it a cheap fix instead of an architecture change. The implementer corrected its own probe's
wording when it caught this and kept the first version beside the fix.

### Why it was not fixed in C-110

Two separate reasons, and only the first is about scope:

1. **Closing it needs no gold access**, so it is *not* blocked by C-110's stop condition — but
   `RESULT.txt` is a **live pipeline artifact** and its seam belongs to **F-148 / C-111**, not to an
   acceptance-instrument card.
2. **Closing it does not fix the headline defect.** Even with both fields printed,
   `RESULT: FAIL` **stays wrong on a correct decline**, because the verdict line is computed from
   `status` alone and `result_text` receives *a manifest row and a paper dict — never a `GoldCase`*.
   **Printing more context does not make a wrong verdict right.**

### Disposition

**Routed into C-111** as an in-scope item under "timeout source" — it is the same distinction that
card exists to make visible, at the reporting end rather than the preservation end.

**The verdict-line problem stays OPEN and is not chartered.** Making `RESULT: FAIL` correct on a
decline requires the batch runner to know the paper is a negative control, and **that fact lives only
in the gold set.** Coupling the live pipeline runner to the benchmark gold is an architecture
decision the Lead has explicitly reserved. C-110 reports the correct status in the **acceptance
instrument**, which is where the gold legitimately lives; `RESULT.txt` remains a run-time artifact
that cannot know.

### Standing lesson

**"The information is absent" and "the information is present but unexplained" are different
defects with different costs.** The first needs a new capability; the second needs two print
statements. **Measure which one you have before charging the price of the first.**

---

## F-159 — `failure_kind = contract` does not mean a contract failure; it means "there were issue codes"

- **Severity** MEDIUM · **Class `product_contract_violation`** (a classification that reads as
  evidence and is not) · **Registered 2026-08-31 (ORCH-717)**
- **Surfaced by REV-110** while attacking C-110's condition 2. **Verified by the Lead in shipped
  code before registration.** Not introduced by C-110 — it is a pre-existing property of
  `batch/driver.py` that C-110 was the first consumer to depend on.

### The measurement

`src/t2pw/batch/driver.py::_classify`, verbatim at the integration tip:

```python
if ambiguous:                        return KIND_AMBIGUOUS
if no_reactions:                     return KIND_NO_REACTIONS
if contract_signal or issue_codes:   return KIND_CONTRACT      # <-- before network/llm
...
if network:                          return KIND_NETWORK
if llm:                              return KIND_LLM
```

**`issue_codes` alone returns `KIND_CONTRACT`, and it is tested before the network and LLM markers.**
So a provider failure that happens to carry any issue code is classified `contract`.

Measured against the shipped function:

| input | resulting `failure_kind` |
|---|---|
| network text, no codes | `network` |
| network text, **one code** | **`contract`** |
| llm text, **one code** | **`contract`** |

`_fail` clears no artifacts and **appends** codes, so `files` stays non-empty on such a row — an
artifact-presence check does not catch it either.

### It is not a bug in `_classify`, and that is what makes it dangerous

`_classify`'s own docstring states the rule it is obeying: *"Order matters. Structured evidence beats
wording, and wording beats the mere presence of a traceback."* **Issue codes are structured evidence
and wording is not**, so preferring codes is deliberate and defensible **as a classifier**.

**The defect is in what the label then licenses downstream.** `contract` reads, to any later
consumer, as *"this leg stopped for a declared contractual reason"*. What it actually means is
*"something attached an issue code"*. Those are different claims, and the second is not evidence of
the first.

**`contract` is also the dominant bucket** — 55 legs against `no_reactions`'s 8 in the 27-manifest
survey — so a consumer that trusts it is trusting the largest and least specific class in the set.

### What it cost, and what it will cost again

C-110 read `contract` as a decline and, combined with issue codes independently satisfying its
"stated reason" condition, admitted three casualties as `PASS_NEGATIVE_CONTROL` — including a row
whose shipped message is literally **`"no research report was produced and no reason was given"`**
(`driver.py:2565`). **A message that says no reason was given was scored as a stated reason.**

That is C-110's blocking finding and it is being corrected there. **This finding exists so the next
consumer does not repeat it.**

### Disposition

**Registered, not fixed.** Changing `_classify`'s ordering would move a classification every
downstream reader already depends on, and **no card in this wave owns `batch/driver.py`**. The
narrow fix is in the consumer: **do not read `failure_kind == contract` as evidence of a declared
stop.** C-110 is removing it from its decline allow-list.

**Relevant to T-108 reporting:** any run report that groups or counts by `failure_kind` must not
present `contract` as "stopped for contract reasons" without saying it also absorbs coded provider
failures.

### Standing lesson

**A classifier's precedence order is a statement about how to LABEL, not about what the label
MEANS.** `_classify` prefers structured evidence over wording, which is right. But a consumer reads
the output as a claim about the world, and the further a label travels from its classifier the more
authority it silently acquires. **Before depending on an enum value, read the branch that produces
it** — the name will always sound more specific than the condition.

---

## F-160 — a same-length mutation can be silently NOT EXECUTED, and the harness reports it as SURVIVED

- **Severity** HIGH for the sprint's own instruments · **Class: infrastructure defect** — it
  corrupts *measurements*, not the product · **Registered 2026-08-31 (ORCH-717)**
- **Found by REV-108** when its own re-run of a fixed finding reported a **false GREEN**.
  **Reproduced independently by the Lead** — `evidence/orch717_pyc_staleness.py` / `.log`, G11
  `ORCH-717/29`.

### The measurement

CPython's default bytecode invalidation (PEP 552 *timestamp* mode) keys a `.pyc` on
**(source mtime truncated to whole seconds, source size)**. A mutation that changes **neither** —
any same-length edit landing in the same wall-clock second as the write before it — leaves the
cached `.pyc` looking valid, and **the interpreter runs the OLD bytecode.**

Reproduced from a clean scratch module:

```
1. wrote ORIGINAL, imported            -> VALUE = 'AAA'
2. bytecode cached                     -> victim.cpython-313.pyc exists=True
3. wrote MUTANT (same length)          -> size 15 -> 15, mtime_s unchanged
4. (mtime_s, size) UNCHANGED?          -> True
5. reloaded                            -> VALUE = 'AAA'      <-- the MUTANT never ran
```

REV-108 confirmed it on the real tree: source git-clean with `{0,3}` twice, while the `.pyc`
recorded identical mtime **and** size. Its `{0,3}` → `{0,0}` mutation is length-preserving.

### Why this is worse than an ordinary flaky test

**The failure direction is a false GREEN, and a false green here INVENTS A COVERAGE GAP THAT DOES
NOT EXIST.**

A mutation harness removes a guard and expects a test to go red. If the mutant never executes, the
suite passes and the harness reports **`MUTATION SURVIVED`** — which reads as *"this guard is
protected by no test"*. **The natural response to that report is to weaken or delete the guard, or
to spend a correction round writing a test for coverage that is already there.** The instrument
built to prove a guard is load-bearing can therefore argue for its removal.

**It is intermittent, which makes it worse.** Whether it fires depends on whether the two writes
straddle a second boundary. **A passing re-run looks like confirmation** rather than like a coin
landing the other way.

### Who escaped it, and only by luck

The mutations in this sprint that appended a **marker comment** — `# MUTATION M16`,
`# MUTANT: enforcement removed` — changed the file **size** and were therefore invalidated
correctly. **That is incidental, not designed.** C-108's `N17` is named by REV-108 as escaping for
exactly this reason.

**It also produced a near-miss report.** REV-108 saw three failures running the C-108 test file
alone, and was about to report a split-vs-combined discrepancy — a finding shape this sprint takes
seriously. After clearing `__pycache__` the split run was **211 passed**. **It was contamination,
not a finding, and the reviewer caught it before reporting.**

### Disposition

**Registered. Not chartered as its own card**, because the fix belongs wherever mutations are
applied and two cards are mid-flight there. **The rule, effective immediately:**

> **Clear `__pycache__` between applying a mutation and running the suite** — or guarantee every
> mutation changes the file **size**. **Do not rely on mtime.**
>
> **A `MUTATION SURVIVED` result from a same-length mutation is not a result.** Re-take it with
> caches cleared before recording it, and never act on it.

Relevant to `evidence/c102_mutation_attack.py` (the harness SMOKE gates), `c107_mutation_attack.py`,
C-108's own 18 mutations and C-110's 12. **The ones that changed size are sound; the ones that did
not must be re-taken before they are trusted.**

**Also recorded:** REV-108 deleted tracked `.pyc` files while clearing caches and restored them;
both worktrees verified git-clean afterwards.

### Standing lesson

**An instrument that reports "no coverage here" is making a claim about the world, and it can be
wrong in the direction that costs a guard.** This sprint already knows to distrust a green test
suite. **The harder discipline is distrusting a green MUTATION — because the reassuring reading of
`SURVIVED` is that your tests are weak, and the alarming one, that your mutation never happened, is
the one nobody checks.**

### ADDENDUM to F-159, 2026-08-31 — two more routes, both ZERO-INSTANCE, found by REV-110 round 1

**The same disease one level down: `failure_kind` is a lossy label and its consumers trust it.**
Registered here rather than as new findings, because the mechanism is identical.

**(i) `_NO_REACTION_MARKERS` is also tested before the network/LLM markers.** So a provider casualty
with **no** issue codes and no-reaction *wording* is labelled `no_reactions` — the one label C-110
does still accept as a declared decline. Measured across 27 manifests (T-107 excluded by name,
non-vacuously): `no_reactions` = 8 legs, **all 8 with files, all 8 with zero codes, suspicious = 0**.

**Why this is registerable where round 0's was blocking:** `contract` was 55 legs and needed only
*any* code. This needs a rare conjunction — provider failure, no codes, and no-reaction wording —
and it occurs **nowhere** in the corpus.

**(ii) `driver.py:2217` labels an ACQUISITION failure as `KIND_NO_REACTIONS`** — *"paper has no full
text, so there was nothing to extract"*, `status=fail`. **That is the ruling's missing-artifact
population wearing a decline label.** It is refused today **only** because that path preserves no
artifacts. **With one file preserved it would earn the status.** Zero occurrences.

> **(ii) is the sharpest available evidence that C-110's artifact condition is LOAD-BEARING rather
> than merely strict.** The condition was questioned as over-conservative — 14 of 55 `contract` legs
> pay for it. This is the row it was buying.

---

### ADDENDUM to F-160, 2026-08-31 — the remedy dirties the worktree, and that breaks SMOKE

**Found by REV-110 while applying F-160's own fix.**

**Purging `__pycache__` DELETES TRACKED FILES in this repository.** Measured by the Lead:

```
56 tracked .pyc files:  51 src/__pycache__ · 2 src/tools/__pycache__
                         2 __pycache__      · 1 scripts/__pycache__
.gitignore lists BOTH  __pycache__/  and  *.pyc
```

They predate the ignore rule, and **git tracks what it already tracks regardless of `.gitignore`**.

So F-160's remedy leaves the worktree dirty — and SMOKE contains
`test_c106_mutation_harness_executable.py`, which asserts `git status --porcelain` is clean for the
file it guards. **The fix for one instrument trips another.** REV-110 restored both trees and
verified them clean; **the next agent following F-160 will hit this unless warned.**

**Do not "solve" it by untracking the 56 files.** That is a repository-wide change nobody has
chartered, and `.git` is already 158 MB. **Restore what you purge, and verify clean before SMOKE.**

**A second, general process rule falls out of the same collision**, and both the C-110 author and
REV-110 independently asked for it to be durable:

> **Any card modifying a file that `c102_mutation_attack.py` guards must COMMIT before running
> SMOKE.** `test_04_restore_is_byte_exact_on_the_real_mutated_module` checks `git status
> --porcelain`, not merely that the bytes were restored. An uncommitted edit presents as **502
> passed / 1 failed** — which reads exactly like a regression and is not one. C-110 hit this and
> round 0 passed only because it happened to run after its own commit.

### AMENDMENT to F-160, 2026-08-31 — it is not confined to mutation harnesses. **PYTEST reports a false GREEN.**

**Escalated by the C-108 author in round 2, after correcting its own first measurement.**

The finding was registered as a hazard to *mutation harnesses*. **That understated it.** Measured
with the arms run in isolation:

```
ARM 0  plain import, no purge   ->  returns the OLD value; the edit never executed
ARM 1  pytest,       no purge   ->  220 passed, exit 0   on a tree whose SOURCE SAYS OTHERWISE
ARM 2  pytest,       purged     ->  RED, 2 failed        (the truth)
```

**A stale-bytecode false green can reach ANY suite run**, not just a mutation arm. A green
`503 passed` proves the bytecode that ran was green — **it does not prove the source on disk is
green**, whenever a same-length edit has landed in the same wall-clock second.

### The author's first attempt was WRONG in the reassuring direction, and that is the useful part

It initially measured ARM 1 as RED and **nearly recorded "pytest is immune" as a finding.** The run
was contaminated: the (e) mutation tests apply and restore mutations *to the same file*, so a pytest
warm-up churns that file's mtime and the later `os.utime` moves the source **away** from the cached
key. The cache then missed **for a reason unrelated to the defect**, and the suite went red for the
wrong reason.

> **The probe was perturbed by the very mechanism it was measuring.** Preserved as
> `c108_r2_f160_demo.attempt1-pytest-warmup-churned-mtime.log`. Running ARM 0 first and **alone** is
> what produced the clean measurement.

### CORRECTION to this finding's own remedy — the Lead's instruction was unsafe

F-160 as first written said *"clear `__pycache__`"*. **In this repository that deletes tracked
files.** Both the C-108 author and REV-110 hit it independently; the author's unscoped
`find -name __pycache__ -exec rm -rf` removed **all 56**, and it was caught only by that card's own
pre-commit boundary check, restored from `HEAD` without overwriting, with `git ls-files -d` verified
back to 0.

**The rule, corrected:**

> **Purge only `src/t2pw` and `tests`.** Neither contains tracked `__pycache__`
> (`git ls-files | grep -E "^(src/t2pw|tests)/.*__pycache__"` → 0). **Never purge unscoped**, and
> **never "solve" it by untracking the 56** — that is a repo-wide change nobody has chartered, on a
> `.git` already at 158 MB. **Restore what you purge and verify clean before SMOKE**, which asserts
> `git status --porcelain` on the file the mutation harness guards.

### Standing lesson, revised

The original lesson was *distrust a green mutation*. **The stronger form: a green test run is a
statement about the bytecode that executed, and only a statement about your source if you can show
the two agree.** Every mechanism this sprint uses to check its own work — pytest, the mutation
harnesses, the split runs — reads through that same cache.

---

## F-161 — the gold-readers selection is not a superset of SMOKE, so a gold edit's mandated gate was structurally blind

- **Severity** HIGH for the sprint's own instruments · **Class: defect in the REVIEW INSTRUMENT** —
  the criteria were incomplete; **the reviewer is not at fault** · **Registered 2026-09-01**
- **Written by C-113, held rather than committed** because `card/C-112-residual-sweep` also edited
  this file and was unmerged at the time. **Sequenced in by the Lead after C-112 merged.** That hold
  was correct behaviour and is recorded as such.
- **Found by** the failed merge of F-150 half 1 at `b05a7281`, measured by the Lead's A/B:
  `evidence/orch718_smoke22_postf150.log` / `orch718_smoke22_postrevert.log`, G11 `ORCH-718/04`
  and `/05`.

### The measurement

REV-F150's mandated gate for the gold edit was the **22-file gold-readers selection**. It ran the
four-step A/B honestly and got a **byte-identical `456 passed / 8 skipped / exit 0` in both arms**,
per-file delta zero on all 22 files, and returned **VERIFIED — APPLY HALF 1**.

`tests/test_c102_coverage_denominator.py` **reads the gold** — it builds
`{case.paper_id: case for case in load_gold_set(pinned_gold_set_path()).cases}` at module scope —
and it **is in SMOKE**. It is **not in the gold-readers selection.** So the gate mandated for a gold
edit could not see the only two tests that edit actually moved:

```
WITH the gold edit     b05a7281   501 passed / 2 failed   exit 1
WITHOUT the gold edit  700c9434   503 passed / 0 failed   exit 0
```

Both failures were in that one file. **A real consequence sat exactly one selection away from the
gate written to find it,** and the arms agreed to the test because neither arm ran the file.

### Why this is a defect in the instrument, not in the reviewer

**The reviewer did precisely what its criteria asked, and did it correctly.** Its A/B was sound:
same tree, same interpreter, predictions written first, one failed measurement kept. Its verdict is
still correct — REV-F150 is not reopened, and the gold edit re-landed unchanged at C-113 with the
byte-identical blob `36f4b7b6…`. What failed is the **choice of population** the criteria named as
sufficient.

**This is the exact mirror of the standing lesson that SMOKE does not cover the gold readers.** The
sprint already knew one direction of that gap. The other direction is just as real:

> **Neither selection is a superset of the other. A gold edit needs BOTH.**

### The consequence, stated plainly

A gate that is *mandated* rather than *chosen* carries the authority of the process. When such a
gate is blind by construction, a green result from it is read as a licence to merge — and it was:
the merge went in, SMOKE caught it, and merge rule 10 required the merge not to stand. **Merge rule
10 was the only thing between an instrument gap and a landed regression-shaped tip.**

### Disposition — RATIFIED BY THE LEAD, 2026-09-01

**Effective immediately, for any change to `src/t2pw/bench/gold/pinned_v1.json`:** run **BOTH** the
22-file gold-readers selection **and** SMOKE, and report both, each with the gold SHA it was
measured against. **Neither alone certifies a gold edit.**

**On the authority of this clause.** REV-113 registered, correctly, that a card had declared a
sprint-wide process obligation on its own authority (its R13), and asked the Lead to ratify or
reword it. **The Lead ratifies it as written and adopts it as a standing obligation.** The
distinction matters and is preserved rather than smoothed over: C-113 was right on the merits and
out of its authority to bind the sprint, and a card proposing a rule it cannot itself enact is the
correct behaviour — the reviewer catching that it had not been ratified is what closes the loop.

**Also adopted, from REV-113's R12:** whenever a census pin in
`tests/test_c102_coverage_denominator.py` moves, close the mutation gap with a **same-length**
mutation of the moved literal. `assert withheld == 100` is unreachable in either red arm — the
set-equality above it aborts the test first — so the natural revert gives that pin **no mutation
coverage at all**, and only a deliberate same-length mutation (the F-160 trap, walked into on
purpose) proves the assert executes.

**RAISED, NOT ANSWERED — still open for the product owner / a future card:** should the
gold-readers selection be *extended* to include every SMOKE file that reads the gold, starting with
`tests/test_c102_coverage_denominator.py`? That is a **`TEST_MATRIX` change with its own cost** —
runtime, per-branch obligations, and a moved `456 / 8` baseline that every future A/B is measured
against. **It was not C-113's to make and it is not settled by this ratification.** The standing
obligation above makes the gap safe without deciding it.

---

## F-162 — a mistyped task id did not return nothing; it returned ANOTHER task's evidence

- **Severity** HIGH at base, **FIXED by C-112** as a side effect of the same guard ·
  **Class** false-PASS in the evidence-reachability instrument · **Registered 2026-09-01 by the Lead**
- **Found by** the C-112 author's own **failed assertion**, and preserved as
  `evidence/c112_r2_false_pass_vectors.attempt1-typo-assertion.log`. **Given a finding id at
  REV-112's request (its R3), which observed it was documented only in a probe source and a test
  docstring.**

### The measurement

C-112's charter described the third R2 vector as *"`--allow-empty` plus a mistyped task id is a
silent pass"* — i.e. exit 0 on nothing. **The truth at base is worse.** `rev109` and `REV_109` did
**not** enumerate zero artifacts and exit 0. They enumerated **REV-109's actual evidence** and
returned **1**, because the filesystem is case-insensitive and `task_stem` strips non-alphanumeric
characters, so three distinct spellings collapse onto one real task directory.

**A typo did not fail loudly, and it did not fail quietly either — it silently certified one task's
work using another task's evidence.**

### Why the charter understated it

The charter reasoned from `--allow-empty` disarming the exit-3 protection, which is a real and
separate vector. It did not anticipate that the id would *resolve* rather than *miss*. **The author
found it only because it asserted the charter's version and the assertion failed** — the failed
measurement was the finding.

### Blast radius — verified nil

REV-112 checked every committed G11 report in the tree: **no report anywhere carries a
non-conforming task id**, so no past certification was mis-attributed. The defect was reachable but
never taken.

### Disposition

**Fixed in C-112**, in the same hunk as the vector it was found under: the grammar guard now refuses
both spellings at exit 2, before enumeration. **No further action.** Registered because a defect
found, fixed and blast-radius-checked should be findable by id and not only by reading a probe.

---

## F-163 — `HeavyLock.release` is not atomic, and its window creates a lock nobody is permitted to clear

- **Severity** HIGH for wave throughput · **Class** orchestration-tooling defect ·
  **NOT REPAIRED — deliberately. Chartered for the next wave.** · **Registered 2026-09-01 by the Lead**
- Full incident record: `evidence/orch718_anonymous_heavy_lock_incident.md`. First reported by the
  C-111 agent as its R-C111-4.

### The defect

`HeavyLock.release` **unlinks `holder.json`, then `rmdir`s the directory**
(`bounded_run.py:828-860`). `acquire` is `os.mkdir`. A kill landing **between those two statements**
leaves an **empty directory with no holder file**, so every subsequent job gets
`BOUNDED_RUN_HEAVY_LOCK_HELD` (exit 95) **forever**.

**The standing clearing checklist is then unsatisfiable by construction**, because its first
condition is *multiple byte-identical holder samples* and there is no holder. **An anonymous lock is
unclearable even by its own owner.**

### The diagnosis is provable, not inferred

If the wrapper's `finally` had simply never run, **`holder.json` would still be there.** It was gone
*and* the directory remained. That state is reachable only by a kill inside `release`, between the
two statements.

### How it was recovered, and the technique worth keeping

**The holder sample survived in a `BOUNDED_RUN_HEAVY_LOCK_HELD` diagnostic**: a failed acquire
prints the holder file verbatim, so an earlier exit-95 in a transcript or committed log **is** the
byte-exact sample the checklist demands. The *source* was stripped; the evidence was not. That plus
a dead PID (473280), three stable empty samples, zero sprint-owned Python processes and confirmed
peer non-ownership satisfied the checklist honestly. **Cleared with `rmdir`, never `rm -rf`** —
`rmdir` refuses on a non-empty directory, so a race that wrote a holder between the last sample and
the call fails the clear instead of being silently overwritten.

**A subagent must not clear an anonymous lock.** *"There is no holder, so nothing owns it, so I may
delete it"* is right by luck and wrong by method — the same reasoning deletes a lock during another
job's acquisition window. The C-111 agent declined and escalated, which is why this was a clean
recovery rather than a guess.

### Why it was not fixed this wave

`bounded_run.py` is the instrument every job in flight is measured through, and **its build hash is
recorded in every G11 report** (`sha256:83d1395…`). **Changing the instrument mid-wave would break
comparability across reports already written.**

### The related cost, same wave, different statement

Three further strands came from the **sibling** failure mode: a shell clock shorter than the
wrapper's `--timeout`, which kills the wrapper so its `finally` never runs at all. **The Bash tool
caps a single call at 600 s**, so a `--timeout 900` wrapper can *never* be covered by a foreground
shell — and a lock-wait retry loop wrapped around a long job blows the cap on **waiting alone**.
Under this wave's contention, acquiring took **19, 15, 120 and 330 attempts** on different jobs.

**Standing rule, adopted:** keep the wrapper `--timeout` comfortably **under** the shell cap, and
never let a lock-wait retry loop share the job's budget. A gate that must queue behind other agents
belongs in **tracked background under D-026**, which is precisely the judgement the C-111 agent made
for its gold-readers gate and which REV-111 endorsed.

---

## F-164 — the recursion fix that closed a false PASS opened a false FAIL, via the allocator's own `.staging`

- **Severity** MEDIUM · **Class** orchestration-tooling defect, introduced by C-112's R2 vector-1 fix ·
  **NOT REPAIRED — chartered for the next wave** · **Found by the Lead at integration, 2026-09-01**
- **Neither C-112 nor REV-112 could have seen it**, and that is the point: it is only reachable by
  running the check **across all four of this wave's reviewer worktrees at once**, which is an
  integration action, not a card action.

### The measurement

Closing out the wave, `evidence/reviewer_evidence_route.py` was run for all four reviews:

```
REV-111   enumerated 14   reachable 14   unreachable 0   all_reachable      exit 0
REV-113   enumerated 16   reachable 16   unreachable 0   all_reachable      exit 0
F-150     enumerated  9   reachable  9   unreachable 0   all_reachable      exit 0
REV-112   enumerated 17   reachable 12   unreachable 5   unreachable_evidence exit 1
```

**All five "unreachable" files are inside `evidence/g11/REV-112/.staging/`** — the allocator's
**reservation** directory, not evidence:

```
UNREACHABLE [unreachable_absent] g11_report: .../g11/REV-112/.staging/03-rev112-c107-mutation.json
UNREACHABLE [unreachable_absent] g11_report: .../g11/REV-112/.staging/04-rev112-route-split.json
UNREACHABLE [unreachable_absent] g11_report: .../g11/REV-112/.staging/06-rev112-smoke.json
UNREACHABLE [unreachable_absent] g11_report: .../g11/REV-112/.staging/07-rev112-goldreaders.json
UNREACHABLE [unreachable_absent] g11_report: .../g11/REV-112/.staging/08-rev112-own-driftscan.json
```

### REV-112's evidence is COMPLETE. The verdict is wrong, not the evidence.

**21 REV-112 files are on integration** — all **10** final G11 reports `01`…`10`, **9** logs and
**2** probe sources. **Nothing is missing.** `.staging` is **untracked and has never been tracked**:
`git ls-files … | grep -c '\.staging/'` returns **0** repo-wide, and the directory does not exist on
integration at all. The checker is comparing transient scratch files against a tree that is
*correctly* not carrying them.

### Why it is C-112's fix, and why it fires on exactly one task

C-112's R2 vector 1 — **the vector that mattered** — was that *probes in a subdirectory are never
enumerated* because the glob was non-recursive and `is_file()` dropped the directory silently. The
fix recurses. **The recursion now also descends into the allocator's dot-prefixed `.staging`
directory**, whose *contents* are ordinary non-dot `.json` files even though its *name* is dotted.
The old `iter_reports` rule — *"every non-dot file in a task folder"* — protected the folder level
only, and recursion moved past it.

**Why only REV-112:** every reviewer worktree has a `.staging` directory, but only REV-112 has files
left in it.

```
rev111 : staging_dirs=1  staging_files=0
rev113 : staging_dirs=1  staging_files=0
revf150: staging_dirs=1  staging_files=0
rev112 : staging_dirs=1  staging_files=5
```

**A reservation is staged before the job runs; a job that never starts leaves it behind.** REV-112
reported **seven `exit 95` lock-held events**, the most of any job this wave. **So the two dominant
infrastructure phenomena of this wave are the same phenomenon here** — F-163's lock contention
manufactures the leftovers that F-164's recursion then misreads.

### Direction of the failure, and why it is MEDIUM not HIGH

**It fails safe.** C-112 replaced a **false PASS** (evidence missing, check green) with, in this
corner, a **false FAIL** (evidence present, check red). A false FAIL costs an investigation; a false
PASS retires the manual habit that works. **The trade is still strongly positive and the fix
stands.** But a check that goes red on correct evidence will be *disbelieved*, and a disbelieved
check is on its way back to being no check.

### Disposition — NOT fixed this wave

The obvious repair is to skip dot-prefixed **directories** during the walk, restoring the folder
rule at every level, and it is one line. **It is not taken here** for the same reason `bounded_run.py`
was not repaired for F-163: this is an instrument the wave's own certifications were produced
through, it has just been independently reviewed, and changing it after its review — without a new
review — is precisely the move this sprint refuses.

**Chartered for the next wave.** Any card taking it must:
- skip dot-prefixed directories at **every** level, not just the task folder;
- **prove the C-112 vector stays closed** — `rev109_probes/deep_probe.py` must still be enumerated
  (a fix that re-breaks recursion to silence `.staging` would restore the original false PASS);
- add a case with a **populated** `.staging` and assert `all_reachable`;
- and consider whether `.staging` leftovers should be cleaned on exit-95, which is F-163's territory
  and may be the better root fix.

**Until it lands:** a `unreachable_evidence` verdict naming only `.staging/` paths is **this defect,
not missing evidence**. Confirm by checking that the task's final reports are present on integration
— and do not let that shortcut become a habit of dismissing red route checks.
