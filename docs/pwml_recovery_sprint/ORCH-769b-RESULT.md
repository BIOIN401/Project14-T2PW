# ORCH-769b: RESULT (quarantine checks an actor against the final gate's protein registry; chlorophyll 9 × unknown modifier)

| | |
|---|---|
| Lane | `orch769b/actor-bucket-quarantine`, worktree `.claude/worktrees/orch769b`, D-112 § 3(b) |
| Base | `e3d5b65f` (= `main` = `sprint/pwml-recovery`) |
| Commits | `76e2d5f2` test-first · `a3ebd555` code · docs commit (this file, evidence) |
| Status | **Implemented and tested. Not reviewed, not merged, not pushed.** It needs an independent review of the diff. |
| Classification | `product_contract_violation`. A `review_required` pathway got no PWML (merge rule 7; PRODUCT_CONTRACT "a valid pathway core suppressed …") |
| Reports | G11 `evidence/g11/ORCH-769b/01`–`19`, logs `evidence/orch769b/NN-*.log`. Every report shows `FINAL SURVIVING COUNT : 0`. No `--heavy-lock` was used, and every job ran in the foreground. |
| D-112 § 2 | **ACCEPT.** PWML 51/91 → **52/91**. Target fixed. No passing leg turned failing, and no gold recall was lost (§ 5). |

## 1. The defect

The leg is ORCH-767 chlorophyll-strict (PMC5130987, `2026-10-23_2328`), and the diagnosis is
`evidence/orch769_diag/769b/` in the primary checkout.

1. Stage 1 declared PIF1, GBP, FAR1, CPO1 and PPO1 twice: as `proteins` and as `nucleic_acids`. The audit
   deleted the five protein rows. Its guard pools every bucket (`apply_audit_patch.py:861,882-903`), so the
   surviving gene rows hid the loss (`03-guard.out.txt`). That is F3 in the diagnosis, and it is not this lane.
2. Quarantine's `_actor_verdict` looked each actor up in the five-bucket registry (`_build_registry`).
   `PIF1` resolved to its gene row, and `_entity_representability` returns True for `nucleic_acids`. So all
   32 processes were admitted.
3. The final gate checks enzymes and modifiers against `proteins ∪ protein_complexes` only. It raised
   9 × "Unknown protein/modifier reference" (PIF1, GBP ×3, FAR1, CPO1 ×2, PPO1 ×2), and no PWML was written
   for a `review_required` pathway.

In this leg all five names are declared **only** as `nucleic_acids`: `01-fixture.log` greps every bucket by
name and synonym.

## 2. The predicate, mirrored from the gate

The gate is `process_normalizer.run_strict_post_normalization_gates`, at base line numbers:

| gate | file:line | what it does |
|---|---|---|
| registry | `process_normalizer.py:4639` | `protein_registry_norms = _entity_name_norms(proteins) \| _entity_name_norms(complexes)`: names **and** synonyms, `_normalize`d |
| reactions | `:4961-4978` | `enzymes` and `modifiers`. **Dict actors only** (`:4967` `if not isinstance(actor, dict): continue`); name `_actor_name_from_row`; an empty name is skipped; `_normalize(name) not in protein_registry_norms` raises "Unknown protein/modifier reference" (`:4973-4978`) |
| transports | `:4980-4999` | `transporters`, by the same rule (dict only `:4988`, check `:4994`), raises "Unknown transporter reference" |
| not read | the gate body (`:4606-5137`) | `reaction_coupled_transports`, interactions and sub_pathways actors: the function has no reference to any of them |

**Compound modifiers.** The registry holds no compounds. So a compound named as an enzyme or modifier fails
the gate, as does a cofactor such as PLP. `_catalyst_reference_norms`' docstring (`:856-863`) says the same.
The rule therefore quarantines compound modifiers too (test `test_g9_a_compound_named_as_a_modifier_*`).

**Transporters.** These are mirrored, with the same rule and its own reason (`non_protein_actor_in_transporters`).

**What the rule mirrors exactly:**

* the registry is built with the gate's own `_entity_name_norms`, over the same two lists. `strict_quarantine`
  already imports `_entity_name_norms`, `_normalize` and `_actor_name_from_row` from `process_normalizer`;
* the name comes from the same `_actor_name_from_row`;
* only dict actors are judged;
* the slots are the same.

**What it deliberately does not do:**

* **Bare-string actors are skipped**, because the gate skips them. A string actor is refused later by the
  stage contract's `actor_schema_not_canonical` (`stage_contracts.py:812`), unchanged. The preservation test
  pins this.
* **No explicit `reaction_coupled_transports` exclusion.** `_ACTOR_SLOTS` lists RCT, but no RCT row ever
  reaches `_actor_verdict`. Admission quarantines every RCT first (`RCT_NOT_EXPORTABLE_REASON`), and closure
  revalidates accepted rows only. A guard would be dead code, so the docstring says it instead.
* **The resolved bucket is not used.** `registry.resolve()` is first-wins across buckets. A name can resolve
  to a compound (compounds register first) while being a protein synonym, and the gate accepts that name.
  Diagnosis counterfactual B (`replay_767.py`) keyed on the resolved bucket. It agrees on this leg but is
  wider than the gate. `test_preservation_the_predicate_is_the_gate_registry_not_the_resolved_bucket` pins
  the difference.

## 3. The diff (`src/t2pw/pipeline/strict_quarantine.py`, +24 −3; nothing else in `src/`)

* `_Registry.protein_norms: Set[str]`: a new field, documented as the gate's registry.
* `_build_registry`: fills it with `_entity_name_norms(proteins) | _entity_name_norms(protein_complexes)`.
  The registry is rebuilt after every closure prune (`quarantine_and_close`, `:2551`), so revalidation sees
  the pruned registry. The diagnosis's F1′ note warns that norms computed once from the input would not.
* `_actor_verdict`: after the existing `unresolved_actor_in_<slot>` check and before representability, a
  dict actor whose `_normalize(name)` is not in `registry.protein_norms` returns
  `(QUARANTINED_BROKEN_REFERENCE, "non_protein_actor_in_<slot>", name)`. The docstring is corrected: it
  claimed the gate aborts on "an actor naming nothing declared".
* One function serves both callers: admission (`_admit_processes`, `:1702`) and closure revalidation
  (`_revalidate_surviving_processes`, `:2102`, recorded as `stranded_by_closure:non_protein_actor_in_<slot>`).

**Untouched:** every gate (D-106); `apply_audit_patch.py` (F3); the actor rows themselves (F2: the rule
quarantines the process and never drops the actor); `streamlit_app.py`; prompts and LLM inputs;
`QUARANTINE_POLICY_VERSION` (§ 8, item 3). **ORCH-766** edits the same module, but its hunks start at lines
27, 419, 2034 and 2533 and this lane's sit at 630, 676 and 1340-1383. A trial merge,
`git merge-tree --write-tree a3ebd555 orch766/drop-unreferenced-species` (head `040d65b9`), is clean:
exit 0, tree `97d2ea82…`, no conflict.

**No LLM-bound change.** Quarantine decides which rows are exported. It writes no prompt. ORCH-759c F5
established that its one in-loop caller, the RAG round's `_classification`, writes a channel that nothing
reads.

## 4. G9: base RED → tip GREEN

Fixture `tests/fixtures/orch769b/` holds two files:

* the leg's `final_mapped.json` **verbatim**, gzipped. The test asserts its sha256 `5805a848…4f8c`. It is the
  canonical checkpoint: it hashes to the leg's `resulting_payload_hash` `sha256:d6fa0ff69ec9aaf9`;
* `coverage.requested_context` from the leg's `quarantine_report.json` (sha256 `df1d9680…`).

Both were built by `evidence/orch769b/make_fixture.py` (`01`).

| test (`tests/test_orch769b_actor_protein_registry.py`) | base `76e2d5f2` (`03-g9-red`) | tip `a3ebd555` (`05-g9-green`) |
|---|---|---|
| `test_g9_leg_quarantines_exactly_the_five_reactions_whose_actor_is_no_protein` | **FAIL**: quarantined `[]` | pass |
| `test_g9_leg_final_gate_passes_on_the_quarantined_payload` | **FAIL**: 9 gate errors, the first `('/processes/reactions/17/modifiers/1', 'Unknown protein/modifier reference: PIF1')` | pass, 0 errors |
| `test_g9_a_modifier_declared_only_as_a_gene_is_quarantined` | **FAIL**: `core_accepted` | pass |
| `test_g9_a_compound_named_as_a_modifier_is_quarantined` | **FAIL**: `core_accepted` | pass |
| `test_g9_a_transporter_declared_only_as_a_gene_is_quarantined` | **FAIL**: `core_accepted` | pass |
| `test_g9_closure_revalidation_applies_the_same_predicate` | **FAIL**: `core_accepted` | pass, `stranded_by_closure:non_protein_actor_in_modifiers` |
| 4 × `test_preservation_*`: protein-synonym modifier, complex-named modifier, compound-first/protein-synonym shadow, bare-string actor | pass | pass |

Every failure is on a value. Every name the module imports exists at base with the same signature.

**What the leg test pins at tip:**

* the same `decision_input_hash` `sha256:6c5f9c8b2410d8af`;
* exactly the five quarantines, all `quarantined_broken_reference`:
  * r17 `GluTR (HEMA1) reaction`: `modifiers`, PIF1;
  * r18 `ALAD (HEMB1) reaction`: `modifiers`, FAR1;
  * r21 `Glu-tRNAGlu reduction to GSA`: `enzymes`, GBP;
  * r22 `Coprogen III to Protogen IX`: `enzymes`, CPO1;
  * r23 `Protogen IX to Proto IX`: `enzymes`, PPO1;
* counts 17/10/0/0/0/5/0 (base 20/12/0…), with `quarantined_disconnected` 0;
* `ok`, no refusal, `review_required`, and `missing_anchors` `["POR (PORB)", "GLK1"]` (base `["POR (PORB)"]`;
  see F3);
* 22 reactions and 5 interactions.

## 5. D-112 § 2: the corpus measurement (ORCH-768 harness, read-only from its worktree)

Both trees are `git archive` exports in scratch: base `e3d5b65f` and tip `a3ebd555`. Each was run with
`orch768_replay_corpus.py run`, offline, under `bounded_run.py`.

| | base (`13-corpus-base`) | tip (`14-corpus-tip`) |
|---|---|---|
| PWML, strict legs with an outcome | **51/91** | **52/91** |
| PWML, replayed legs | 51/81 | 52/81 |
| tier A / tier B | 8/28 / 43/63 | 9/28 / 43/63 |
| replayable failures | 30 | 29 |
| network attempts (probe ok, 3 counted) | 0 | 0 |

The base JSON is **byte-identical** to ORCH-768's committed baseline `orch768_corpus_baseline.json`, sha256
`0e17b69a…` for both. That makes it an independent repeat of the baseline.

`compare base tip --target ORCH-767/chlorophyll-strict/2026-10-23_2328` (`15-corpus-compare`):

```
pass rate base 51/91 -> tip 52/91
fixed: ['ORCH-767/chlorophyll-strict/2026-10-23_2328']
passing legs turned failing: none
gold recall lost: none
still failing, at a different step (information): ['runs_verify/2026-08-22_2017/PMC12856317: final_pre_export_stage3_gates -> refinement_review_crash']
target ORCH-767/chlorophyll-strict/2026-10-23_2328: FIXED
D-112 § 2: ACCEPT
```

**Gold** (certified gold; enterobactin and menaquinone only, no chlorophyll gold). All 17 gold rows are
identical at base and tip, at both final and PWML level (`corpus-base.txt` / `corpus-tip.txt`, GOLD section).
For example, 767 menaquinone PWML P 0.909 R 1.000 F1 0.952 and 756 enterobactin P 0.714 R 1.000 F1 0.833.

**Blast radius, field by field** (`evidence/orch769b/corpus_delta.py`, `16-corpus-delta`). Of 93 replay
records, **90 are identical, including every written file's sha256**. Three differ:

| leg | base | tip |
|---|---|---|
| ORCH-767 chlorophyll 2328 (target) | final gate, 9 × unknown modifier | **PWML** `pathway.review_required.pwml` (sha256 `5b40c6ec…`), final gate ok, required-field gate ok, IR clean, release `review_required` |
| runs_verify 2026-08-22_2017 PMC12856317 (heme) | final gate, "Unknown protein/modifier reference: Pyridoxal 5'-phosphate" | final gate ok. The only reaction is quarantined, so 0 reactions and 2 interactions remain, then `refinement_review_crash` "DraftGraph has no reaction nodes". No PWML on either side (F2) |
| runs_verify 2026-08-24_1203 PMC12856317 (heme) | quarantine refusal `unexportable_entity:2` | the same refusal; one reaction more is quarantined (PLP again). Outcome unchanged |

**Why no passing leg can turn failing.** Consider a leg whose final gate passed at base. Every dict actor of
every surviving process was in the final payload's protein registry. Closure only removes rows, so at every
admission or closure step the registry is a superset of the final one, and the rule never fires on a process
that survived at base. The corpus agrees: 0 legs broke, and 90/93 records are unchanged.

## 6. Gates

| # | gate | report | result |
|---|---|---|---|
| 1 | G9 RED at test-first `76e2d5f2` (base `src`) | `03-g9-red` | 6 failed (the `test_g9_*`), 4 passed (preservation) |
| 1 | G9 GREEN at `a3ebd555` | `05-g9-green` | **10 passed** |
| 1 | corpus compare | `13`, `14`, `15` | **ACCEPT**, 51/91 → 52/91 |
| 2 | affected: 28 files. They are every test file matching `strict_quarantine\|quarantine_and_close\|freeze_canonical_payload\|run_quarantine_boundary\|write_quarantine_artifacts\|_build_registry\|entity_type_overlap\|_prune_entities`, plus `test_stage3_gate_report_lifecycle.py`. This includes all `test_strict_quarantine*.py`, `test_orch759c_*`, `test_c011_*` and `test_stage3_gate_report_lifecycle.py`. AppTest files go to gate 5 and Chunk E is run separately | `06-affected` | **609 passed, 8 skipped**, 0 failed. The 8 skips are the known data-driven skips in `test_strict_failure_replay.py:234` (2) and `:249` (6) (`18`) |
| 2 | Chunk E `test_strict_quarantine_real_artifact_replay.py` | `07-chunk-e` | **175 passed**, no pin moved |
| 3 | SMOKE | `08-smoke` | **508 passed** |
| 4 | `tests/test_eval_v2_*.py` (6 files) | `09-eval-v2` | 89 passed, 1 xfailed |
| 5 | `test_streamlit_quarantine_boundary.py` alone | `10` | 23 passed |
| 5 | `test_c052_prefreeze_report_at_the_streamlit_seams.py` alone | `11` | 13 passed |
| 5 | `test_c055_rag_loop_wiring.py` alone | `12` | 14 passed |

Chunk D was not run, as instructed. There were no failures to explain. Every pytest run went through
`pinned_pytest.py`, with `PYTHONPATH=<worktree>/src` and a fresh `--basetemp` under `C:/t/bt-orch769b/`. The
pins are in `evidence/g11/pin/ORCH-769b/`.

## 7. Findings

* **F1. The defect** (§ 1). Quarantine called an actor usable when it was declared in **any** graph bucket.
  The final gate accepts proteins and complexes only. The two disagree on any actor that is declared only as a
  gene, a compound or an element collection.
* **F2. A compound cofactor costs its whole reaction.** On both heme legs (PMC12856317), the extractor put the
  cofactor `Pyridoxal 5'-phosphate` (`entity_type: compound`, `role: cofactor`) in the modifiers of
  `ALAS2-catalyzed ALA formation` (glycine + succinyl-CoA → aminolevulinic acid). That is the leg's only
  reaction. At base the final gate refused it, so no PWML was written. At tip the reaction is quarantined and
  the graph keeps 0 reactions and 2 interactions. The refinement review then crashes with the ORCH-769c defect
  (`DraftGraph has no reaction nodes`). The harness's informational
  `export_if_refinement_opened ok=True, required_gate_codes []` predicts that this leg would write an
  interactions-only PWML once 769c lands. That pathway would be correct but missing its core reaction.
  Nothing passing broke, but the cost is real (§ 9, item 1; `17-moved-leg-probe`).
* **F3. The chlorophyll cost.** Each of the five quarantined reactions keeps a surviving reaction with the
  same inputs and outputs, so no conversion step leaves the graph. The catalysts do not all survive with it
  (`19-twin-probe`, `evidence/orch769b/twin_probe.py`):

  | quarantined | surviving twin | catalyst on the twin |
  |---|---|---|
  | r17 `GluTR (HEMA1) reaction`, r21 `Glu-tRNAGlu reduction to GSA` | `GluTR reaction` | `GluTR1 complex` (the same enzyme as HEMA1) |
  | r18 `ALAD (HEMB1) reaction` | `ALA dehydratase reaction` | **none**. The verified `Delta-aminolevulinic acid dehydratase 1, chloroplastic` is lost |
  | r22 `Coprogen III to Protogen IX` | `Coprogen III oxidase reaction` | **none**. `Coproporphyrinogen-III oxidase 1, chloroplastic` is lost |
  | r23 `Protogen IX to Proto IX` | `Protogen IX oxidase reaction` | **none**. `PPO` is lost |

  Three steps are therefore exported **uncatalysed**. The PWML also loses the regulators that only r17 and r21
  reached: `HY5`, `PIF3`, `GLK1`, `GLK2`, `FHY3`, `FLU`, `ARR10`, `ARR12`, with their wrappers (diagnosis B
  lists all 28 entity rows removed, `01-replay.out.txt`). The anchor `GLK1` goes with them, and completeness
  drops from 0.933 to 0.867. Each of these reactions leaves because **one** of its actors is a gene row: r22
  and r23 name the right enzyme beside a gene-only duplicate (`CPO1`, `PPO1`). That is the price of
  quarantining the whole process rather than the dangling actor (diagnosis F2, a product choice; § 9 item 1).
* **F4. Blast radius is 3 of 93 replay records** (§ 5), and no leg that wrote a PWML changed at all.
* **F5. `QUARANTINE_POLICY_VERSION` was not bumped.** The module's own comment (`:213-216`) says to bump it
  "whenever a change in this module can change a verdict for an unchanged payload". This change does, as did
  C-010, C-067, C-121 and ORCH-759c, none of which bumped it. C-114 and C-116 forbid touching it because it
  moves every decision id in the corpus. Left unchanged here, so the `decision_input_hash` matches the leg
  (`sha256:6c5f9c8b2410d8af`, pinned). See § 9, item 3.
* **F6. Research mode** runs quarantine diagnostically and applies nothing. A `non_protein_actor_in_<slot>`
  decision there appears only as a review flag.
* **F7. ORCH-768's acceptance test needs no `EXPECTED` move.** Its three fixture legs (ORCH-767 sour taste,
  enterobactin, WNT PCP) are among the 90 unchanged records. ORCH-768 § 11, item 6 anticipated that 769
  lanes might have to move it. This one does not.

## 8. Failures and dead ends

* **Considered and rejected: key on the resolved bucket** (diagnosis counterfactual B). It gives the same 5 on
  this leg. It is wider than the gate whenever first-wins resolution lands a protein synonym on a compound, so
  it would quarantine reactions the gate passes. The mirrored predicate is the exact one (§ 2).
* **Considered and rejected: compute the protein norms once from the input** (diagnosis F1′,
  `replay_767_gatepred.py`). Closure revalidation runs against the pruned registry, so the norms live on the
  registry and are rebuilt with it. The closure test exercises this.
* **Considered and rejected: an explicit `bucket in ("reactions", "transports")` guard.** No RCT row can
  reach `_actor_verdict` (§ 2), so the guard would be dead code. It is documented instead.
* `02-draft-base` and `04-draft-tip` are draft runs of the new test file before each commit. After `02`, the
  leg gate test was changed to return the gate's full error list rather than raise. That makes the RED output
  show the 9 errors (`03`).
* The first corpus probe (`16`) showed only the counts for the heme leg. `17-moved-leg-probe`
  (`evidence/orch769b/moved_leg_probe.py`) replays the two heme legs to name the quarantined reaction and the
  PLP modifier.

## 9. NOT settled (for the product owner)

1. **Whole-process quarantine vs dropping only the dangling actor (diagnosis F2).** Today's cost:
   * chlorophyll exports three steps uncatalysed (ALAD, CPO, PPO) and loses 8 regulators and the GLK1 anchor
     (F3);
   * heme loses its only reaction to a cofactor (F2).

   Options:
   * (a) **accept** (this lane). Cost: what is listed above. No edit of extracted content; incomplete pathways
     are preserved as `review_required`.
   * (b) **drop the non-protein actor and keep the reaction, flagged for review.** Cost: quarantine starts
     editing a row's actor list, which is the "silent edit" the module was built not to make. That needs a
     PO ruling and its own corpus run.
   * (c) **re-file compound cofactors upstream** (normalizer: a `role: cofactor` compound modifier becomes a
     reaction participant or a review note). Cost: a Stage-3 lane, which ORCH-768 F-6 shows the corpus
     cannot measure, plus a decision on how PWML should show a cofactor.
2. **The audit guard that hid the loss (diagnosis F3,** `apply_audit_patch.py:861,882-903`**).** The guard
   pools buckets, so deleting a protein row is allowed while a same-named gene survives. This lane makes the
   consequence safe (the process leaves). It does not stop the deletion. Options: make the guard per-bucket
   (cost: an LLM-stage change, measurable only live), or leave it.
3. **`QUARANTINE_POLICY_VERSION`** (F5). Options:
   * (a) **leave it** (this lane). Cost: a stored decision from before this change would match the new rules'
     hash. No production path reuses one across a code change today.
   * (b) **bump it.** Cost: every `decision_input_hash` in the corpus moves (81/81 fidelity becomes "changed"),
     along with pinned tests: this lane's `6c5f9c8b…`, ORCH-759c's `d2ef87f2…`, and the C-114 checks.

   A one-off bump for every verdict-changing lane since `2026-07-31.1` is the third option.
4. **Bare-string actors** stay outside the rule because the gate does not read them. The stage contract
   refuses them later (`actor_schema_not_canonical`). Quarantining them too would be wider than the gate.
   Is that the wanted boundary?
5. **The heme leg now depends on ORCH-769c.** Once 769c renders an empty-reaction graph, the 2026-08-22_2017
   heme leg would write an interactions-only PWML. Whether that is wanted is item 1's question.
6. **Not confirmed live.** No pipeline run was allowed. ORCH-770 is the live confirmation (D-112 § 4).

## 10. TEST_MATRIX delta (proposed; this lane does not edit TEST_MATRIX.md)

Append at end of file:

```markdown
## ORCH-769b: test obligations, appended end-of-file by rule · 2026-10-24

Lane `orch769b/actor-bucket-quarantine` (`76e2d5f2` test-first, `a3ebd555` code). Reports
`evidence/g11/ORCH-769b/01`–`19`.

**SMOKE is unchanged at 508** (`g11/ORCH-769b/08`). Merge rule 10 unchanged. No chunk count moves.

### New focused selection

`tests/test_orch769b_actor_protein_registry.py` (10, ~1.5 s), fixture `tests/fixtures/orch769b/` (the
ORCH-767 chlorophyll-strict leg's `final_mapped.json`, gzipped, sha256 asserted, plus its requested
context). In no chunk and not in SMOKE. G9: 6 `test_g9_*` red at base (`g11/ORCH-769b/03`), 4
`test_preservation_*` green there; 10 green at tip (`05`). Run it for any change to `_actor_verdict`,
`_build_registry`/`_Registry` in `strict_quarantine.py`, or to the actor checks of
`process_normalizer.run_strict_post_normalization_gates`.

### Pinned tests moved (merge rule 4)

None. Chunk E 175 passed unchanged (`07`).

### Known failures

None new. Affected sweep, 28 files: 609 passed, 8 skipped (data-driven skips in
`test_strict_failure_replay.py`) (`06`, `18`).
```

## 11. Evidence index

| artifact | what |
|---|---|
| `evidence/orch769b/make_fixture.py` → `01-fixture.log` | builds the fixture from the leg, prints the hashes and the five reactions' actors |
| `evidence/orch769b/corpus-base.{json,txt}`, `corpus-tip.{json,txt}` | the corpus records (`13`, `14`); base JSON = ORCH-768 baseline, sha256 `0e17b69a6c91e9ac…`; tip JSON sha256 `4c3374894bfd4c48…` |
| `15-corpus-compare.log` | the D-112 § 2 verdict |
| `evidence/orch769b/corpus_delta.py` → `16-corpus-delta.log` | every differing replay field, per leg |
| `evidence/orch769b/moved_leg_probe.py` → `17-moved-leg-probe.log` | the two heme legs at tip: what is quarantined and why |
| `18-skips-strict-failure-replay.log` | the 8 skip reasons |
| `evidence/orch769b/twin_probe.py` → `19-twin-probe.log` | F3: each quarantined chlorophyll reaction's surviving twin and its catalysts (tip canonical payload from the corpus run) |
| `tests/fixtures/orch769b/README.md` | fixture provenance and checksums |
