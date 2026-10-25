# ORCH-772i — integration gates for ORCH-772 + ORCH-769d on `00eb12d3`. RESULT

**Sub-orchestrator record (D-114).** Merged into `sprint/pwml-recovery` with `--no-ff`: `a3b95065` (ORCH-772 @ `34a02fc5`) and
`00eb12d3` (ORCH-769d @ `8a139290`), each after an independent `pwml-reviewer` round 0 APPROVE WITH NOTES whose G11 reports and
review text were first committed to the lane branch (`34a02fc5`, `8a139290`; G11-R reachability).

## Gates (clean worktree `.claude/worktrees/integ772` at `00eb12d3`, no `.env`, heavy lock `ORCH-772i`, `--basetemp` per run)

| gate | report | result |
|---|---|---|
| SMOKE (22 files) | `evidence/g11/ORCH-772i/01-smoke.json` | 511 passed in 46.04 s |
| affected (reaction evidence, ORCH-772 coverage, ORCH-768 corpus test, 13 applier files + ORCH-769d, completeness audit) | `02-affected.json` | 718 passed |
| `tests/test_eval_v2_*.py` | `03-eval-v2.json` | 89 passed, 1 xfailed |
| `test_reaction_evidence_view.py` alone | `04` | 11 passed |
| `test_reaction_evidence_app_hook.py` alone | `05` | 1 passed |
| `test_orch759b_recorder_reattach.py` alone (LLM-input golden) | `06` | 7 passed |

Every report: exit reason `completed`, FINAL SURVIVING COUNT 0, lock acquired and released; free RAM >= 11.7 GB before each job.
Orphan sweep after the gates: no sprint `python.exe` alive (only VS Code `lsp_server.py`). Chunk D not run (final integration only).

## Process deviation (caused by main, recorded at main's request)

Main committed a docs-only DECISIONS entry (`c53f62c9`) on top of the local sprint and pushed, which published the two merges to
`origin/sprint/pwml-recovery` **before** these gates finished. `main` was untouched (`b16a45e7`). The gates then passed, so no
fix-forward was needed. Main has since stopped pushing; the sub-orchestrator pushes main's docs commits with its next gated push.

## Not settled

- ORCH-769d reviewer N1: `_rebase_op` runs before the per-op `try`, so a token `str.isdigit()` accepts but `int()` rejects (`"²"`)
  raises after a shift where base records "Application failed". Theoretical; one-line follow-up inside the applier.
- ORCH-769d `/-` on a missing key creates a dict (RESULT § 7.1); needs a ruling and its own lane (reviewer prefers fail closed).
- ORCH-772 reviewer note 1 (no-PWML legs render `unclassified`) and note 2 (runner `--src`) are in lane ORCH-772b.
