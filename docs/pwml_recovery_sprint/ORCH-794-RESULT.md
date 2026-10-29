# ORCH-794 RESULT: one name declared as two kinds keeps the paper's kind (D-119 § 3, ORCH-789 C7)

Branch `orch794/fix` off `f3a80e09`. Code commit `483f4098` (shared with ORCH-795; see `ORCH-795-RESULT.md`). Not merged, not pushed.
**Merge rule 5 is waived by the product owner (D-119 § 4: no reviewer round for ORCH-790..795).**

## 1. Cause (verified against the ORCH-785 artifacts)

- **chlorophyll** (`orch785run/.../chlorophyll-strict/2026-10-28_1743/.../strict`): `quarantine_report.json` quarantines 4
  reactions `ambiguous_entity_type:compounds+nucleic_acids`: `GluTR1 reaction`, `GluTR reaction`, `Glu-tRNAGlu reduction to GSA`
  (participant `Glu-tRNAGlu`) and `Glu-tRNAGlu formation` (`tRNAGlu`). `merged_payload.json` declares both names as a compound
  and as a nucleic acid; neither kind has a database identity (the run's id cache: `db::trnaglu`, `db::glutrnaglu` are
  `novel_compound`). The closure then removed all four rows as unreferenced (`removed_entity_report.json`).
- **wnt_ca** (`.../wnt_ca-strict/2026-10-28_1910/...`): 3 transports and `IP3 binds InsP3R` are quarantined
  `ambiguous_entity_type:compounds+proteins` on `InsP3R`. The compound row is not in `merged_payload.json` (the audit says it
  removed it, `audit_iteration_summary.json`), yet the quarantine input has it, with an `__auto_state__` compound location:
  a later step re-declared it. Which step is not pinned here (outside this lane). The protein `InsP3R` has no UniProt id.

## 2. Change

| file | function | change |
|---|---|---|
| `src/t2pw/pipeline/strict_quarantine.py` | `_keep_one_kind_per_name` (new), `_paper_kind`, `_has_db_identity` (new) | a name declared in exactly two kinds (not nucleic acid + protein: `_split_gene_protein_names` owns that; not two protein kinds) keeps, in order: (1) the only kind with a database identity; (2) the protein kind when the paper writes the name only in protein-only roles (enzyme, transporter, complex component) and never as a reaction participant or transport cargo; (3) the kind its own name says (`tRNA`/`RNA`/`DNA` -> nucleic acid; receptor, channel, transporter, pump, kinase -> protein). The other row and its own-kind location rows are removed and recorded, reason `same_name_declared_as_another_kind:<kept bucket>`, `detail.decided_by`. Nothing decides it -> both rows stay, and admission omits each process naming it per row as before (`ambiguous_entity_type:<a>+<b>`, KI-011) |
| same | `quarantine_and_close` | one call after `_split_gene_protein_names` (its call position is untouched), before the registry; the removals enter `removed_entity_report` at iteration 0. Research mode still returns the input untouched and clears removals |
| `src/t2pw/app/reaction_evidence.py`, `PWML_KNOWN_ISSUES.md` appendix | `KNOWN_ISSUES` | new code row `same_name_declared_as_another_kind:<bucket>` -> KI-051; the `ambiguous_entity_type` row notes the new remaining cases |
| `tests/test_orch794_one_name_two_kinds.py` | new | 5 G9 + 1 preservation |

No gate changed. `_split_gene_protein_names` not edited. Pre-freeze only.

## 3. Proof

| what | tree | result | G11 report (`evidence/g11/ORCH-794/`) |
|---|---|---|---|
| G9 `test_orch794_one_name_two_kinds.py` | base `C:/t/orch794/base` (`f3a80e09`) | **5 failed, 1 passed** (all `test_g9_*` on values; preservation passes) | `26-g9-base-final.json` |
| same | tip `483f4098` | **6 passed** | `25-g9-tip-final.json` |
| directly affected + every non-AppTest importer of `strict_quarantine` / `reaction_evidence` / the corpus harness (39 files) | tip | **967 passed, 8 skipped, 1 failed**: `test_paper_eligibility_corrections::test_the_2122_evaluation_with_cached_abstracts_beats_title_only`, which fails the same on base (pre-existing, unrelated) | `27-affected-final.json`; base `24-eligibility-base.json` |
| AppTest, each alone: `test_reaction_evidence_view`, `test_reaction_evidence_app_hook`, `test_streamlit_quarantine_boundary` | tip | 11, 1, 24 passed | `28-`, `29-`, `30-` |
| leg replay (reconstruction, § 4) | base / tip | below | `31-replay-base-final.json`, `32-replay-tip-final.json`; output `evidence/orch794_replay.txt` |
| `g11_evidence.py check --task ORCH-794` | | exit 0, 16 pinned | `33-g11-check.json` |

## 4. Leg replay: rows recovered

Both legs exported, so their `final_mapped.json` is the canonical payload and no longer holds the rows at issue. The script
(`evidence/orch794_replay.py`) therefore reconstructs the quarantine input for those rows only: each quarantined row back from
its verbatim `original_process`; the entity rows closure removed under the names they reach, from `merged_payload.json`
(Stage-6 fields absent), a single-protein wrapper rebuilt in the leg's own wrapper shape, and `HEMA1`'s accession P42804 from
the run's own id cache (its removal record says it had one); the leg's stale lock entries dropped. Then `replay_leg`, offline.

| leg | rows put back | base | tip | headline (base -> tip, this lane's rows only) |
|---|---|---|---|---|
| chlorophyll | 4 reactions | all 4 `ambiguous_entity_type:compounds+nucleic_acids` | **3 drawn** (`GluTR1 reaction`, `Glu-tRNAGlu reduction to GSA`, `Glu-tRNAGlu formation`); `GluTR reaction` -> `protein_complex_missing_species` on its enzyme `GluTR`, a Stage-6 family complex the leg did not keep (rebuilt by name only: **reconstruction limit, not measurable offline**, KI-058) | 43/82 -> 46/82 (with ORCH-795's denominator: 46/68) |
| wnt_ca | 3 transports, 1 interaction | all 4 `ambiguous_entity_type:compounds+proteins` | **0 drawn**: compound `InsP3R` removed (`protein_only_role`); all 4 now omitted as `protein_missing_external_identity` (KI-001): `InsP3R` has no UniProt id | 10/39 unchanged |

The brief expected chl +4, wnt_ca +3. chl: +3 measured, the fourth is unmeasurable offline. wnt_ca: the dual kind is fixed, but
the rows meet the next defect (§ 8 item 2).

## 5. Pins moved (merge rule 4), exact deltas

- `test_orch759c_quarantine_bucket_closure.py`, chlorophyll fixture: counts core 26 -> 27, broken-reference 10 -> 9; admissions
  sha `8cbc88d2…` -> `53355ece…`; removed-entity rest (25, `b995f2cf…`) -> (23, `54084dfd…`); removed-location rest (43, `2ef23a5a…`)
  -> (41, `eda2b5d1…`); nucleic acids 20 -> 21; reactions 30 -> 31. Cause: `tRNAGlu` keeps its nucleic acid, `Glu-tRNAGlu reduction
  to GSA` is admitted. Values from `10-pins759c-tip.json` (probe `C:/t/orch794/pins759c.py`).
- same file, N4 ×2: the shadow row's reason `unreferenced_after_quarantine` -> `same_name_declared_as_another_kind:proteins` (same
  row removed, one step earlier). N2 shape 1: `glutathione synthesis` `quarantined_disconnected`/`stranded_by_closure:<r>` ->
  `quarantined_unmapped_entity`/`<r>` (same reaction omitted, for the same component, at admission).
- Overlap-gate vehicles `test_strict_quarantine.py::test_an_entity_type_overlap_that_survives_closure_blocks_the_export` and
  `test_strict_quarantine_locks_and_scope.py::test_each_refusal_class_gets_its_own_reason`: the compound now carries a ChEBI id,
  so nothing decides the kind and the overlap gate still refuses. Assertion unchanged (as ORCH-759c did).
- `test_orch772_pwml_coverage.py`: appendix 84 -> 86 codes (this row + ORCH-795's); new label case
  `same_name_declared_as_another_kind:nucleic_acids` -> KI-051 (NEW acceptance).

## 6. Failures, dead ends, wrong theories

1. **Wrong theory: the ORCH-759c fixture declares `Glu-tRNAGlu` twice.** Only `tRNAGlu` (`02-g9-tip-r1`, 1 failed). Test fixed.
2. **Preservation test red at base** (`04-g9-base`): base quarantines every other row of that payload, so nothing survives. Split
   into a G9 test and a preservation test with only the undecidable name.
3. **`gate.sh` with report tree `.`** wrote the base arm's pin verdict into the base tree (it `cd`s there). Moved into this
   worktree, base cleaned. Absolute paths since.
4. **The first role rule was wrong.** It kept the protein complex for `OPC-8:0-CoA`, written as its own reaction's substrate and
   catalyst (`test_strict_failure_replay`, `22-importers-tip`), which would have admitted a metabolite as an enzyme. The role
   rule now abstains when the name is also a reaction participant or cargo.
5. **Replay reconstruction r1-r4** (`12-`..`18-`): rebuilding rows from `detail.row` digests (elided strings), missing Stage-6
   wrappers, and the leg's stale `quarantined_locked_reactions` entries inflated the tip total (108 -> 110). Fixed by the verbatim
   `original_process` and dropping the stale entries (`19-`/`20-`, final `31-`/`32-`).
6. `05-`, `06-`, `11-`: lock held by another job (rc 95, nothing ran).
7. Rule breach, owned: one `python3 -` call with empty stdin (no script, no output). No Python job ran through it.

## 7. KI delta

- **New:** `same_name_declared_as_another_kind:<bucket>` -> **KI-051** (an entity of the wrong kind). Appendix row added. The KI-051
  entry's body (symptom, status, papers) still describes RyhB only; the catalogue owner should widen it to D-119 § 3.
- Unchanged: an undecided dual kind stays `ambiguous_entity_type:<a>+<b>` -> KI-011.

## 8. LLM-side

- Stage 1 declares `tRNAGlu` / `Glu-tRNAGlu` in two kinds (chlorophyll, ORCH-759 and ORCH-785 draws).

## 9. NOT settled (for the product owner)

1. **Which kind a charged tRNA is.** The rule keeps the nucleic acid, which is what the name says (D-119 § 3 "wording"). PathBank
   and KEGG model `L-Glutamyl-tRNA(Glu)` as a compound (KEGG C02987). Options: (a) keep (nucleic acid needs no database id in
   PathWhiz); (b) compound for a reaction participant. Cost of (b): one more rule, and the compound has no PathBank match on this leg
   (`novel_compound`).
2. **wnt_ca's 3 InsP3R transports are still omitted**, now for KI-001 (no UniProt for `InsP3R`; "InsP3Rs" is the ITPR1-3 family).
   Two routes, neither in this lane: (a) apply D-114 AMENDMENT 1 § 2 at admission, keeping the transport and dropping the
   transporter. That is the open ORCH-773 § R1.7 item 1. Gain: wnt_ca +3. Cost: it changes base-passing PWMLs. (b) Add `InsP3R` /
   `IP3R` to the D-119 § 1 family terms (ORCH-790's lane). Gain: the 3 transports with an ITPR member, plus `IP3 binds InsP3R`
   (unsigned, so not in the headline).
3. **`GluTR reaction`** needs a live leg: its enzyme `GluTR` is a Stage-6 family complex, not replayable offline.
4. The wording lists are name-local and small (`ponytail:` note in the code). Widen them only when a leg needs it.

## 10. Proposed LEDGER row

`| ORCH-794 | D-119 § 3 one name in two kinds | orch794/fix @ 483f4098 | built, tested, replayed; rule 5 waived (PO) | G9 26 red / 25 green; affected 967 passed (1 pre-existing); replay chl +3 of 4 (1 unmeasurable offline), wnt_ca +0 (now KI-001, § 9.2) | ORCH-794-RESULT.md |`

## 11. ORCH-794b: regression on the live ORCH-796 chlorophyll leg, fixed

Branch `orch794b/fix`, created at the lead's instruction from the merged tip `55fd55ed`. Code commit `a2762049`. Rule 5 waived
(D-119 § 4).

**Regression (verified, leg `orch796run/.../chlorophyll-strict/2026-10-29_1213/.../strict`, read-only).** No PWML was written:
IR `unresolved_entity_reference`, "Process member 'tRNAGlu' was not found", at `/processes/reactions/18/inputs/1`
(`pwml_ir_report.json`). `_keep_one_kind_per_name` removed compound `tRNA^Glu` and kept nucleic acid `tRNA^Glu`
(`removed_entity_report.json`, `name_wording`). But a reaction input and a `nucleic_acid_locations` row still wrote `tRNAGlu`.
This module's name key (`process_normalizer._normalize`) makes `tRNA^Glu` and `tRNAGlu` one name, so admission accepted the
reaction. The exporter's key (`ir._norm`, `^` becomes a space) makes them two names. At `f3a80e09` both rows stayed, every such
process was omitted per row, and the PWML was written. ORCH-794 turned 4 per-row omissions into a whole-pathway failure.
**My § 4 replay missed it:** its reconstructed legs spell every reference exactly as the kept row.

**Fix:**
- After a decision, new `_spell_references_as` rewrites every process, location and complex-component reference with that
  name key to the kept row's exact name. Free-text fields (evidence, relationship, states, …) are never touched.
- Location rows that become identical collapse into one.
- **The lead's rule 1** (a substrate, product or cargo keeps a kind the IR resolves in that role) already holds: the IR resolves
  reaction members and transport cargo in every kind (`ir.py` `preferred_order["reaction_member"]`; transport elements use the
  same role). So the wording decision stands, and no compound preference was added.
- **The lead's rule 2:** every reference is rewritten to the kept name by construction, so no unresolved spelling is left for a
  fallback to catch.

| proof | result | G11 (`evidence/g11/ORCH-794/`) |
|---|---|---|
| G9 `tests/test_orch794b_spelling.py` (the leg's shape), base = `55fd55ed` content (fix stashed in this worktree) | **1 failed** on `['ir:unresolved_entity_reference'] == []`, preservation passes | `35-b-g9-base-1.json` |
| same, tip | **2 passed** | `34-b-g9-tip-1.json` |
| ORCH-794 set + strict-quarantine files (13 files) | **392 passed, 8 skipped** | `38-b-affected-1.json` |
| leg tail replay (`evidence/orch794b_replay.py`, output `evidence/orch794b_replay.txt`). The input is the leg's payload plus exactly the rows the leg records as removed by this function | base: **no PWML**, `unresolved_entity_reference`, coverage 0/74. Tip: **PWML written** (`review_required`), 54/74 full count; `Glu-tRNAGlu formation` now reads `tRNA^Glu` | `36-b-replay-base-1.json`, `37-b-replay-tip-1.json` |

**Not settled:** the two name keys still disagree everywhere else. A single row `tRNA^Glu` referenced as `tRNAGlu`, with no
second kind involved, would pass admission and fail the IR in the same way. That case predates ORCH-794 and was not seen on a
leg. Options:
- (a) apply the same rewrite to every reference that resolves only under the looser key. Cost: it touches all legs and needs a
  corpus run.
- (b) leave it.
