# ORCH-769b fixtures

Input of the strict quarantine on the ORCH-767 chlorophyll-strict leg, used by
`tests/test_orch769b_actor_protein_registry.py` (the G9 replay).

| file | what it is | source |
|---|---|---|
| `chlorophyll_PMC5130987_767_final_mapped.json.gz` | the leg's `final_mapped.json`, **verbatim**, gzipped (level 9, mtime 0) | `.claude/worktrees/orch767/runs_papers/ORCH-767/legs/chlorophyll-strict/2026-10-23_2328/papers/PMC5130987/strict/final_mapped.json` (untracked run directory) |
| `chlorophyll_PMC5130987_767_requested_context.json` | the Stage-0 context the live quarantine received | `coverage.requested_context` of that leg's `quarantine_report.json` (sha256 `df1d968047352354792d23ce97faa0dc7757d62b068f0d0803aedb713970ddd5`) |

- Uncompressed sha256 of `final_mapped.json`: `5805a848ac81e4eafcd7f12d401f63fb63aaf84d3168f75c7174ed9fd3484f8c`
  (920 593 bytes; gzip 81 088 bytes). The test asserts it, so the fixture cannot drift from the leg.
- It is a **canonical** checkpoint: it hashes to the leg's `resulting_payload_hash`
  (`sha256:d6fa0ff69ec9aaf9`), and replayed at base with `strict_db=True`, `mode="pathwhiz"` and this
  context it reproduces the live decision (`decision_input_hash sha256:6c5f9c8b2410d8af`, 32 processes
  admitted, the final gate's 9 errors exactly):
  `C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/docs/pwml_recovery_sprint/evidence/orch769_diag/769b/01-replay.out.txt`.
- Built by `docs/pwml_recovery_sprint/evidence/orch769b/make_fixture.py`
  (`evidence/orch769b/01-fixture.log`, G11 `evidence/g11/ORCH-769b/01-fixture.json`). No field was edited
  or dropped.
