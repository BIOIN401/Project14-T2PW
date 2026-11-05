# Product-owner rulings, 2026-11-04

Recorded by the orchestrator from the PO's answers in session. **DECISIONS.md is PO-only**, so
these are not appended there. The PO should copy them into `DECISIONS.md` as D-W11 … D-W14.
Until then, this file is the citation for the tasks that rely on them.

| ref | question | ruling | relied on by |
|---|---|---|---|
| Q-R0-1 | Merge gate 5 allows a golden-seam test edit only if it is "only an import path". About 20 tests read `streamlit_app.py` **source** by file path. | **Allow the path repoint.** Pointing a source-reading test (or a patch target) at the new file counts as an import-path edit. The reviewer must confirm that every test edit changes only a path or a patch target, and nothing else. For `test_c011` the BASE side keeps the old path and the tip side uses the new one. | R1 and every later R-task |
| FastAPI | W6 needs FastAPI (named in PLAN §2, not installed). | **Yes: W6 adds FastAPI** to `requirements.txt` and installs it into `.venv`. | W6, W9 |
| Q-W4-1 | D-W07 says to retry "infrastructure failures and no-PWML outcomes". Read literally, that would retry strict runs where a biology gate refused export. | **The narrow rule is confirmed.** A no-PWML retry happens only when `status == error`, the kind is `""` or `unknown`, there is no `pwml_path` and no `release_status`. Gate refusals and research-mode successes are final, and any result carrying a `release_status` is final. This is what is merged at `090d4855`. | W4 (merged) |
| Q-W0-3 / Q-W0-5 | W7 needs to know which consented results go to the curator queue. The citation report exists only in research mode. | **Release-ready PWMLs with provenance.** A strict PWML with `release_status.status == release_ready` goes in, with its provenance reports, plus the citation report when one exists. Review-required results do not go in. No extra research-mode run is made. | W7 |

## Still open (not answered in this round)

Q-W0-2 (deadline form); W2 species scope (4,018 vs the 1,127 with pathways); Q-W3-1..6;
Q-W4-2..5; Q-W8-1..7; W5 branding, and squash vs `--no-ff` for binary-heavy branches; hosting;
the server DB user; the SMTP provider.

## Addendum, 2026-11-05 (R1 review)

| ref | question | ruling | relied on by |
|---|---|---|---|
| R1-F1 | After the move, the golden-seam test `test_streamlit_quarantine_boundary` parses only `service/postpipeline.py`. A future unguarded quarantine call in `streamlit_app.py` would no longer fail it. | **Fix: the test parses both files** (`APP_PATH` and `SERVICE_PATH`), restoring its original coverage. This is the one permitted golden-seam edit beyond a path, and it must be reviewed. | R1 |
| R1-F6 | Under `streamlit run`, the moved code logs as the literal `t2pw.app.streamlit_app` rather than `__main__` (3 `logger.debug` lines, dropped at the default WARNING level). | **Accept the literal name.** It is documented in R1.md and needs no added code. | R1 |
