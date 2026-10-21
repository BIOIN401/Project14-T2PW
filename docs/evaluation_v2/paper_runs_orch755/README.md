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
