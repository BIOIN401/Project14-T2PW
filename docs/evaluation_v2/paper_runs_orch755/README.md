# ORCH-755 paper runs — inputs

The first full-pipeline paper runs in the D-104 production configuration, for the product owner's
hand comparison against the papers.

- `topics/<condition>.txt`: one pinned paper per condition, `PMCID | pathway | organism`, taken
  verbatim from the ORCH-751 cohort's `C:/t/cohort/*/index.json`. Seven conditions on six papers:
  WNT PCP and Wnt/Ca2+ share PMC10425814. Enterobactin (PMC12096016) and menaquinone (PMC12421875)
  are the only two with certified gold.
- Configuration: `.env` sets `T2PW_EXTRACTION_DRAWS=3` and `OPENROUTER_EXTRACTION_MAX_TOKENS=64000`; the
  production prompt is `r2_01_control_v03` (D-104 § 5); the curator is live. Code is the commit `night.log`
  records at start.
- Runner: `docs/pwml_recovery_sprint/evidence/orch755_paper_runs.sh`; review gathering:
  `docs/pwml_recovery_sprint/evidence/orch755_review_pack.py`.
- Outputs are **not committed** (run scratch, like `runs_verify/`): `runs_papers/ORCH-755/`. The human entry
  point is `runs_papers/ORCH-755/REVIEW/INDEX.txt`. Process reports are committed under
  `docs/pwml_recovery_sprint/evidence/g11/ORCH-755/`.

A staging dry run (`--stage-only`, zero LLM legs) passed for all seven before launch: every paper was fetched
in full (36,169 to 200,898 characters), and nothing was skipped (`g11/ORCH-755/01`–`07`).

## ORCH-798: five new papers (2026-10-30)

Five conditions never run before (checked against every PMCID in the repo-root `topics_*.txt`, this folder, the ORCH-724
exclusion set and every `runs*/` directory: `C:/t/orch797/excl.txt`, 227 ids). Picked from Europe PMC open-access search for
diversity; full-text sizes are Europe PMC XML estimates without the reference list:
`staphyloxanthin` PMC12924280 (bacterial metabolic, ~106k chars), `glucosinolate` PMC5116641 (plant, ~107k),
`cgas_sting` PMC13586583 (human signalling, ~113k), `serine` PMC6073169 (human metabolic, ~16k),
`ergosterol` PMC7397035 (yeast, ~44k). Run: `runs_papers/ORCH-798/`; record `ORCH-798-RESULT.md`.
