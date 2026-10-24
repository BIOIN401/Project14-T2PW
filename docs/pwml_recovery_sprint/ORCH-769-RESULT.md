# ORCH-769 — the four ORCH-767 failures, diagnosed and fixed against the replay corpus (D-112 § 3)

**Status:** (b) and (c) MERGED; (a) and (d) diagnosed, fixes waiting on product-owner rulings. Sub-orchestrator record.
Lane records: `ORCH-769b-RESULT.md`, `ORCH-769c-RESULT.md`. Diagnoses (read-only, helpers and outputs):
`evidence/orch769_diag/{769a,769b,769c,769d}/`. Measured with the ORCH-768 replay corpus (`evidence/orch768_replay_corpus.py`,
D-112 § 2 acceptance rule via `compare`).

## Corpus pass rate (strict legs with a PWML written, of 91)

| tree | rate | note |
|---|---|---|
| `main` @ `e3d5b65f` (ORCH-768 baseline) | 51/91 (51/81 replayed) | 30 replayable failures, 10 LLM-originating |
| + ORCH-766 (`19a6a6ce`) | 53/91 | both 3-draw menaquinone legs fixed (first measured fix) |
| + ORCH-768 (`72d04d00`) | 53/91 | harness only |
| + ORCH-769b (`509fe185`) | 54/91 | chlorophyll 767 fixed |
| + ORCH-769c (`ec800370`, Chunk D follow-up `40122c63`) | **56/91** | WNT PCP 767 fixed; `runs_verify/2026-08-22_2017/PMC12856317` too (769b made it interactions-only, which hit the same crash) |

Every step ACCEPT under D-112 § 2: no passing leg turned failing, no gold recall lost (each lane's `compare`, re-run by its
reviewer).

## (a) enterobactin — DIAGNOSED, fix needs a ruling

Stage 2 inferred `enterobactin synthase complex` (evidence not in the paper); `relocate_complex_named_proteins` made a
generated wrapper whose only component is a new row `enterobactin synthase`; the audit nulled its default stoichiometry
(`audit_json_llm.py:46` prompt rule); Stage 6 mapped the row to EntE's accession P10378; the final gate requires a positive
stoichiometry on generated wrappers and refused. Not related to 759a or 759e. Replayable from `final_mapped.json`; the shape
originates in LLM stages. Proposed fix (a product choice, a new removal rule): quarantine an inferred complex-named wrapper
when the reaction keeps an extracted catalyst (gold stays 5/5). Open: should the gate keep requiring a positive count on
generated wrappers (PathWhiz allows null)?

## (b) chlorophyll — MERGED `509fe185`

The audit deleted five protein rows that also existed as gene rows (its guard pools buckets); quarantine's actor check used
a four-bucket registry, the final gate proteins ∪ complexes only. Fix: quarantine judges enzymes/modifiers/transporters by
the gate's exact predicate (`non_protein_actor_in_<slot>`). Review APPROVE WITH NOTES (23 actor kinds × 3 slots, 0 leaks, 0
overreach). Cost: ALAD/CPO/PPO export uncatalysed, GLK1 and eight regulators lost (completeness 0.933 → 0.867); the
recovery is the per-bucket audit guard (open). Heme legs' ALAS2 quarantined for a compound PLP modifier (no PWML before or after).

## (c) WNT PCP — MERGED `ec800370` (+ Chunk D partition move `40122c63`)

An interactions-only pathway (0 reactions, 18 interactions) passed quarantine; the review diagram raised "DraftGraph has no
reaction nodes", the review state was never set, the PWML button never rendered. Fix: render returns `b""` when there is
nothing to draw (every caller tolerates it); the driver reports the canonical payload's counts. Review APPROVE WITH NOTES.
SMOKE 508 → 511. Open (F-7): an interactions-only pathway now exports `review_required`; WNT PCP draws 1 of 18 interactions,
PMC12856317 draws none. Decline or export is a product question.

## (d) Wnt/Ca2+ — DIAGNOSED, fix needs a ruling

The audit patch applier (`apply_audit_patch.apply_patch_with_policy`, `:2680-2746`) applies a remove and then later ops by
their pre-patch indices without rebasing, so each synonym after the removed row lands one row later; a cross-row synonym
kept `phosphorylated CAMKII` alive, which has no admitted identity, so quarantine refused (`unexportable_entity:1`).
Replayable. With the cross-row synonyms dropped: `review_required`, gate passes (9 transports, 13 interactions). Proposed fix:
rebase later ops after an accepted remove/insert (or reject them); a second applier defect (`/-` on a missing key creates a
dict) also. Rebase vs reject goes to the product owner.

## Not settled

1. (a) the removal rule and the positive-stoichiometry gate question; (d) rebase vs reject.
2. (c) F-7: export or decline an interactions-only pathway (options: keep exporting; decline when no reaction or transport
   survives; decline only when the IR draws no process).
3. (b) the audit guard that pools buckets (F3) and whether to bump `QUARANTINE_POLICY_VERSION`.
4. Corpus limits (ORCH-768 reviewer): gold is loaded from the measured tree; `PATHBANK_DB_*` must be unset for a run; Stage-6
   fixes are not measurable offline.
