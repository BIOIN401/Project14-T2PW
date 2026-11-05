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

## Addendum 2, 2026-11-05 (R2 review, schedule)

| ref | question | ruling | relied on by |
|---|---|---|---|
| R2-F2 | Since R2 the golden-seam test `test_streamlit_quarantine_boundary` (:773) parses only the app and `service/postpipeline.py`, not the new `service/rag_orchestration.py`. There is no `quarantine_and_close` call in that file today, so the test still passes, but the guard is weaker. | **Widen the parse set to every `src/t2pw/service/*.py`.** A second permitted golden-seam edit beyond a path, like R1-F1. Done as a follow-up task after R2 merges, and reviewed. | R2-F2 follow-up; R3-R6 |
| SCHED-F1 | Driver bridge (BR): the server's run function wraps `batch/driver.run_one` behind the W0 interface, taking R3-R6 off the LOI path. | **Approved.** Dispatched as task BR. A7 swaps back to `run_pathway` only after the equivalence test passes. | BR, A7, W9 |
| Hosting / DB | Where the public server runs and how it reaches PathBank. | **Deferred by the PO**: "I will work out getting access … to the db and all of that stuff later." W9 stays held. Reminder for that request: per CLAUDE.md the server gets its own **read-only** DB user. | W9, B1, D2 |
| Q-BR-1 | When a PMID/PDF/text job has an organism but no pathway name, the bridge refuses it: the benchmarked driver uses the organism only together with a name. The form required the name only in name mode. | **Require the pathway name in every mode** (PO: "do whatever creates the most correct results … the recommended one"). This is a form edit in the BR fix round; the bridge keeps its own refusal. | BR, W6 |
| Q-BR-2 | Name mode: run only the top paper (`count=1`), one run per paper, or concatenate. | **Keep the top paper (`count=1`)**, which matches one benchmark leg. | BR |
| Q-BR-4 | An unsupported request shows "did not pass the quality checks". | **Specific message** keyed on the bridge's unsupported-request issue code; the generic text stays for real gate failures. | BR, W6 |
| Q-R25-1 | The equivalence test's canonical hash differs every run: `mapping/enrich_entities.py:141 _utc_now_iso()` stamps 15 `retrieved_at` leaves. The PWML bytes are already identical. | **Freeze the clock in the test** (test-only; no src change). App, app and service must give identical full canonical hashes under the frozen clock. The comparison with the original live run uses the PWML bytes and the rest of the payload. The non-reproducible production hash is recorded as a PO finding, unchanged. | R2.5, R3+ |
| Q-R25-2 | The R2.5 fixture embeds one open-access paper's full text (PMC7059786, 1.2 MB gz). | **Commit it only if the licence is CC-BY or CC0**, verified from an authoritative source. Otherwise one more recording of another licensed case with PWML evidence is authorised. | R2.5 |
