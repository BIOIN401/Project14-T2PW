# ORCH-759c fixtures

Input of the strict quarantine on the ORCH-759 chlorophyll-strict leg, used by
`tests/test_orch759c_quarantine_bucket_closure.py` (the G9 replay).

| file | what it is | source |
|---|---|---|
| `chlorophyll_PMC5130987_final_mapped.json.gz` | the leg's `final_mapped.json`, **verbatim**, gzipped (`gzip -n -9`) | `runs_papers/ORCH-759/legs/chlorophyll-strict/2026-10-22_1231/papers/PMC5130987/strict/final_mapped.json` (untracked run directory) |
| `chlorophyll_PMC5130987_requested_context.json` | the Stage-0 context the live quarantine received | `coverage.requested_context` of that leg's `quarantine_report.json` (sha256 `e0e11de6a2522e6e350ccf230489c17f127ab1d85f22f47185d6956a633d161d`) |

- Uncompressed sha256 of `final_mapped.json`: `20c61fd792a7a7e085fa0e2da409ef3d9f4eec400300ff265673337a86431526`
  (987 256 bytes). The test asserts it, so the fixture cannot drift from the leg.
- Replayed with `strict_db=True`, `mode="pathwhiz"` and this context, the base code
  reproduces the live decision: `decision_input_hash sha256:d2ef87f2ae5da361`, counts
  19/36/14/28, `refusal_reasons ["entity_type_overlap:4"]`, `diagnostic_only`
  (`docs/pwml_recovery_sprint/evidence/orch759c/01-base-facts.log`). The live
  `admitted_payload_hash` (`d33121c9…`) differs from the replay's (`b43803ad…`)
  because `final_mapped.json` is not byte-identical to the in-memory object the app
  hashed; the decision input, which is what the decisions are taken on, is identical.
- Gzipped only to keep the working tree small (76 KB vs 987 KB); no field was edited
  or dropped.
