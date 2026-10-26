# ORCH-767 — overnight run, morning summary

Written by the runner at the end of the run (`evidence/orch770_morning_summary.py`). Legs: `C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch767/runs_papers/ORCH-767/legs`.

## Per paper

PWML coverage = processes drawn in the PWML / processes in the text record (D-114; target 75%), from each leg's directory by `t2pw.app.reaction_evidence`. Headline first: the interactions PathWhiz cannot carry (KI-008) are left out (D-114 AMENDMENT 2); the full count follows. A leg with no PWML names the known issue of its recorded cause.

| paper | PWML | PWML coverage | release | reactions / interactions | min | code |
|---|---|---|---|---|---|---|
| enterobactin | **no** | 0/21 (0%); full count 0/21 (0%); no PWML: KI-012 | review_required | 5 / 8 | 64.0 | `d6849c9e` |
| chlorophyll | **no** | 0/61 (0%); full count 0/61 (0%); no PWML: KI-004 | review_required | 27 / 5 | 74.2 | `d6849c9e` |
| sour_taste_v2 | **yes** | 7/12 (58%); full count 7/12 (58%) | review_required | 0 / 0 | 49.0 | `d6849c9e` |
| wnt_pcp | **no** | 0/66 (0%); full count 0/66 (0%); no PWML: unclassified | review_required | 0 / 18 | 103.0 | `d6849c9e` |
| wnt_ca | **no** | 0/48 (0%); full count 0/48 (0%); no PWML: KI-016 | diagnostic_only | 4 / 33 | 109.2 | `d6849c9e` |
| menaquinone | **yes** | 14/14 (100%); full count 14/14 (100%) | review_required | 11 / 1 | 55.5 | `d6849c9e` |

## Gold accuracy (eval-v2 score_leg)

| paper / level | ORCH-767 |
|---|---|
| enterobactin stage1 | P 1.000 R 1.000 F1 1.000 (5/5) |
| enterobactin final | P 1.000 R 1.000 F1 1.000 (5/5) |
| enterobactin pwml | not written |
| menaquinone stage1 | P 1.000 R 1.000 F1 1.000 (9/9) |
| menaquinone final | P 0.909 R 1.000 F1 0.952 (9/9) |
| menaquinone pwml | P 0.909 R 1.000 F1 0.952 (9/9) |

## Failures, one line each (from each leg's RESULT.txt)

- **enterobactin:** post-pipeline validation failed: 1 blocking issue(s) at final_pre_export_stage3_gates [gate.generated_protein_complex_enterobactin_synthase_complex_componen]
- **chlorophyll:** post-pipeline validation failed: 9 blocking issue(s) at final_pre_export_stage3_gates [gate.unknown_protein_modifier_reference]
- **wnt_pcp:** the app never rendered the "refinement_generate_pwml" button ('refinement_generate_pwml'), so the review step that unlocks PWML export was not reached [(none reported)]
- **wnt_ca:** post-pipeline validation failed: 1 blocking issue(s) at final_pre_export_gate_lifecycle [final_gate_report_missing] [gate.gate_lifecycle]

## Paths

Review copies: `C:\t\orch772b\ms767\REVIEW/<paper>/strict/` (reaction_evidence.txt, RESULT.txt, stage1_ensemble_report.txt, the PWML where written); index `REVIEW/INDEX.txt`.

## LLM health

```
rag_prose_extraction: 0/104 empty | Stage 1 extraction: 1/51 empty | gap resolver: 0/36 empty | Stage 2 inference: 6/34 empty | preprocessor: 0/23 empty | audit loop: 1/16 empty | chat: 0/9 empty
```
```
    LONG/ERROR  Stage 2 inference attempt 1: finish_reason=length chars=0 ~1216 s
    LONG/ERROR  audit loop attempt 1: finish_reason=length chars=0 ~1410 s
    LONG/ERROR  audit loop attempt 2: finish_reason=error chars=0 ~1277 s
TOTAL model attempts 273, long (>= 1200 s) 3, finish_reason=error 1
```
