# evidence/orch803 -- ORCH-803 no-loss scan

- `no_loss_scan.py`: `scan` (run with base src, then tip src, on PYTHONPATH) and `compare`.
- `scan_base.json`, `scan_tip.json`: per-text abbreviation pairs and replayed gate verdicts.
- `scan_compare.json`, `scan_compare.log`: removed/added pairs and verdict flips.
- Bounded-run reports: `../g11/ORCH-803/03-scan-tip`, `04-scan-base`, `05-scan-compare`.
- Classification and limits: `../../ORCH-803-RESULT.md` section 4.
