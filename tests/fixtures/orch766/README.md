# ORCH-766 fixtures

The strict quarantine's output on the ORCH-759p menaquinone leg, used by
`tests/test_orch766_unreferenced_species.py` (the G9 replay, D-111 section 1).

| file | what it is | source |
|---|---|---|
| `menaquinone_PMC12421875_759p_final_mapped.json.gz` | the leg's `final_mapped.json`, **verbatim**, gzipped (level 9, mtime 0) | `runs_papers/ORCH-759p/legs/menaquinone-strict/2026-10-22_2154/papers/PMC12421875/strict/final_mapped.json` (untracked run directory) |
| `menaquinone_PMC12421875_759p_requested_context.json` | the Stage-0 context the live quarantine received | `coverage.requested_context` of that leg's `quarantine_report.json` (sha256 `2d741ed46b421f099fe01a63ab42cd6e74c65920ffa244964da720646fd5c809`) |

- Uncompressed sha256 of `final_mapped.json`: `34a2b0e15a750df396c3178e3f663dce0e2be3938fec49cbd7e1bf2b9a953650`
  (252 353 bytes). The test asserts it, so the fixture cannot drift from the leg.
- `admitted_payload_hash` of it is `sha256:47d5937e59d3da3d`, the leg's own
  `resulting_payload_hash`: this file IS the frozen quarantine output, and replaying the
  quarantine on it at the base SHA `d6849c9e` returns it unchanged (a fixed point;
  `docs/pwml_recovery_sprint/evidence/orch766/` census, base run).
- Species rows: `Lactococcus lactis` (1358), `Lactococcus lactis subsp. lactis KF147`
  (684738, all 7 Men proteins), `L. lactis subsp. lactis KF147` (no taxonomy id, referenced
  by nothing: the row the leg's `pwml_required_field_gate_report.json` refused on) and
  `Arabidopsis thaliana` (the PathBank `Unknown` sentinel protein's species).
- Gzipped only to keep the working tree small (34 KB vs 252 KB); no field was edited or
  dropped.
