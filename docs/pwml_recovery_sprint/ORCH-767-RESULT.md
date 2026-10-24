# ORCH-767 — the overnight run at 3 draws on the merged fixes (D-111 § 2)

**Status:** DONE 2026-10-24 05:59 (summary written late, ~13:50). Sub-orchestrator record. The product owner's one-page summary is
`runs_papers/ORCH-767/MORNING_SUMMARY.md` (run scratch, not committed); its content is reproduced below as the durable record.

Runner: `evidence/orch767_night.sh` (one `orch756_paper_runs.sh` call per paper from worktree `.claude/worktrees/orch767` pinned at
`d6849c9e`; gold scorer after enterobactin and menaquinone). ORCH-759p's legs were scored the same way for comparison
(`evidence/orch759p_gold_accuracy.txt`, G11 `ORCH-767/09`).

## Incident: the summary was 8 hours late

The night driver called `orch767_morning_summary.py` at the end, which had not been written yet (`night.log` `summary_exit=2`); the
orchestrator meant to write it during the night, but its session was blocked ~15 h: a subagent's own Agent dispatches block its tool
round until the child finishes, even with `run_in_background: true` (memory note `unattended-subagents-hang-on-permission-prompts`),
and the ORCH-766 implementer ran ~15 h. Fix (D-112 process rule): every future runner writes its own end-of-run summary in the runner.

Run: 6 strict legs, 3 draws, 10,800 s per leg, from worktree `.claude/worktrees/orch767` pinned at **`d6849c9e`**
(main; all merged lanes through ORCH-759e; NOT ORCH-765/766), `.env` copied from the primary (draws 3, budgets 128000,
extraction 64000). 2026-10-23 22:24 → 2026-10-24 05:59, every report `FINAL SURVIVING COUNT : 0`
(`docs/pwml_recovery_sprint/evidence/g11/ORCH-767/`). The extra `orch767b` menaquinone leg did not run: ORCH-766 had not merged.

## Per paper (code `d6849c9e` for every leg)

| paper | PWML | release | reactions / interactions | min |
|---|---|---|---|---|
| enterobactin | **no** | review_required | 5 / 8 | 64 |
| chlorophyll | **no** | review_required | 27 / 5 | 74 |
| sour taste (*Mus musculus*) | **yes** | review_required | 0 / 0 (transport pathway) | 49 |
| WNT PCP | **no** | review_required | 0 / 18 | 103 |
| Wnt/Ca2+ | **no** | diagnostic_only | 4 / 33 | 109 |
| menaquinone | **yes** | review_required | 11 / 1 | 55 |

## Gold accuracy (eval-v2 `score_leg`; `evidence/orch767_gold_accuracy.txt`, `orch759p_…`, `orch756_…`)

| paper / level | ORCH-767 (3 draws) | ORCH-759p (3 draws) | ORCH-756 (1 draw) |
|---|---|---|---|
| enterobactin Stage 1 | P 1.000 R 1.000 F1 1.000 (5/5) | P 1.000 R 0.800 F1 0.889 | P 1.000 R 1.000 F1 1.000 |
| enterobactin final | P 1.000 R 1.000 F1 1.000 | P 1.000 R 1.000 F1 1.000 | P 0.714 R 1.000 F1 0.833 |
| enterobactin PWML | not written | **P 1.000 R 1.000 F1 1.000** | P 0.714 R 1.000 F1 0.833 |
| menaquinone Stage 1 | P 1.000 R 1.000 F1 1.000 (9/9) | P 1.000 R 1.000 F1 1.000 | P 1.000 R 1.000 F1 1.000 |
| menaquinone final | P 0.909 R 1.000 F1 0.952 | P 1.000 R 1.000 F1 1.000 | P 0.889 R 0.778 F1 0.830 |
| menaquinone PWML | **P 0.909 R 1.000 F1 0.952** (9/9 gold, 11 rows) | not written | P 0.889 R 0.778 F1 0.830 |

Per-paper gold detail: `runs_papers/ORCH-767/REVIEW/{enterobactin,menaquinone}/strict/gold_comparison.txt`.

## Failures, one line each (not yet diagnosed beyond the leg's own RESULT.txt)

- **enterobactin:** `final_pre_export_stage3_gates`, `gate.generated_protein_complex_enterobactin_synthase_complex_componen…`. The Stage-2-inferred "enterobactin synthase complex" wrapper again (the E1 shape), now failing on its component.
- **chlorophyll:** `final_pre_export_stage3_gates`, 9 × `gate.unknown_protein_modifier_reference`. Reaction modifiers name proteins that are not in the payload.
- **WNT PCP:** crash, "the app never rendered the `refinement_generate_pwml` button", so the review step that unlocks the PWML export was not reached.
- **Wnt/Ca2+:** `final_pre_export_gate_lifecycle`, `final_gate_report_missing` (the strict quarantine refused; release `diagnostic_only`).
- **menaquinone:** PWML written. The unreferenced abbreviated species row (D-111) did not appear on these draws, so ORCH-766 was not needed this time.

## Paths (review copies; legs under `.claude/worktrees/orch767/runs_papers/ORCH-767/legs/<paper>-strict/`)

`runs_papers/ORCH-767/REVIEW/<paper>/strict/` holds, for each paper:
- `reaction_evidence.txt`
- `RESULT.txt`
- `stage1_ensemble_report.txt` and `.json`
- `ensemble_identity_merge_report.json`
- the PWML where one was written: `sour_taste_v2/strict/pathway.review_required.pwml` and `menaquinone/strict/pathway.review_required.pwml`

Index: `runs_papers/ORCH-767/REVIEW/INDEX.txt`.

## LLM health (`evidence/orch767_trace_stats.txt`, `orch767_leg_stats.txt`)

- 273 model attempts. Empty `finish_reason=length` completions: Stage 2 6/34, Stage 1 1/51, audit 1/16, all others 0. That is up from 1/309 in ORCH-759.
- 1 provider `finish_reason=error`.
- 3 calls of 20+ minutes:
  - Stage 2, 1,216 s, empty at `length`;
  - audit, 1,410 s, empty at `length`;
  - audit, 1,277 s, `error`.
- The 128k budget is reached by reasoning alone on some calls.

## Not settled

- The four new failure modes above need diagnosis.
- ORCH-765 (round 2 FINAL) and ORCH-766 are implemented and await review; neither is in this run.
