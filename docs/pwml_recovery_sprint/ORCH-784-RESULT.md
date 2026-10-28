# ORCH-784 RESULT — live 6-paper run on the ORCH-782/783 merge

**Code:** `9312999c` (ORCH-782 KI-011 gene/protein split + ORCH-783 KI-001 family complexes, D-117).
**Runner:** `evidence/orch770_night.sh <orch784run> ORCH-784`. Pinned worktree `.claude/worktrees/orch784run` @ `9312999c`. Strict mode, 3 draws, 10,800 s per leg. Started 2026-10-27 19:31, done 2026-10-28 03:02.
**Lifecycle:** G11 reports `evidence/g11/ORCH-784/01`-`08`, every leg exit 0 with 0 survivors. The copied `.env` was deleted when the runner exited, and the heavy lock was released.
**Summary (untracked, runs_*):** `runs_papers/ORCH-784/MORNING_SUMMARY.md`. Gold scores: `evidence/orch784_gold_accuracy.txt`.

## 1. Result

| paper | PWML | coverage (headline) | ORCH-781 |
|---|---|---|---|
| enterobactin | yes | 8/12 (67%) | 7/10 (70%) |
| chlorophyll | **no** | - | 24/82 (29%) |
| sour_taste_v2 | **no** | 0/18 | 6/11 (55%) |
| wnt_pcp | yes | 5/64 (8%) | no PWML (timeout) |
| wnt_ca | **no** | - | 15/53 (28%) |
| menaquinone | yes | 22/25 (88%) | 14/14 (100%) |

Gold F1 (pwml level): enterobactin 0.800 (was 1.000), menaquinone 0.947 (was 1.000).

**The D-117 effect on coverage is not measurable from this run**: both target papers, chlorophyll (KI-011) and Wnt/Ca2+ (KI-001), wrote no PWML.

## 2. Findings: three whole-pathway stops (D-114/D-116 say every pathway leads to some PWML)

1. **wnt_ca: ORCH-783 regression** (`product_contract_violation`). Mapping stopped at the `post_remap` boundary with code `generated_wrapper_component_missing_external_identity`: "Generated wrapper component protein must include a UniProt or DrugBank identifier." The payload held 14 protein complexes. A family-complex member with no id stops the pathway, where it must instead fall back to the Unknown-backed complex or be omitted per row. Evidence: `.claude/worktrees/orch784run/runs_papers/ORCH-784/legs/wnt_ca-strict/2026-10-28_0044/papers/PMC10425814/strict/mapped_failure_snapshot.json` (`merged_payload.json` and `LEG_TRACE.jsonl` sit beside it). The ORCH-783 unit tests did not cover a member that ends up with no id after remap.
2. **chlorophyll: `PREFREEZE_DUPLICATE_CANONICAL_ROWS`.** Canonicalization puts two compound rows, `protoheme` and `heme`, on the one canonical name `heme`, and the pre-freeze check refuses the pathway. No link to ORCH-782 has been shown; a compound alias collision seems likely, but this is not verified.
3. **sour_taste_v2: PWML IR validation failed.** The biological states `Type III taste cell state` and `afferent nerve state` have no resolved subcellular location reference and no `subcellular_location_key`. This is unrelated to D-117.

## 3. Disposition (Lead ruling, 2026-10-28)

Each stop becomes a per-row repair or omission, recorded in the text record with a KI id, never a whole-pathway refusal. A row is merged with an existing one only where the identity is the same, and otherwise omitted. No biological gate is weakened. Each fix is proven offline on that leg's saved payload before any live run. Then only the 3 failed legs are re-run live, under task id ORCH-785.

## 4. Not settled

- Enterobactin and menaquinone gold F1 fell (0.800 and 0.947, from 1.000). Not triaged. Draws vary at temperature 0 (CLAUDE.md), so re-run before calling this a regression.
- WNT PCP now finishes (95 min) but covers only 8%: of the 66 omissions, 36 are KI-001 and 23 are KI-015. Not triaged.
