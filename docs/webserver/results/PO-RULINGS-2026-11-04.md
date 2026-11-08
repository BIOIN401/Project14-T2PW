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

## Addendum 3, 2026-11-05 (live end-to-end runs E2E-1 and E2E-2)

| ref | question | ruling | relied on by |
|---|---|---|---|
| Q-E2E-1 | Worker time budget. Live runs showed one full paper finishing in 35 min (E2E-2) and the same paper running past 60 min (E2E-1) because of LLM latency. History: median 18 min, p90 43 min, 15 of 280 legs over 60 min. As shipped: 60 min per attempt, timeout retried, 2 h cap. | **One 90-minute try. A timeout is not retried.** `RUN_TIMEOUT_S` = 90 min, and the per-item wall cap is also 90 min, so the single worker is held for at most 90 min per pathway. Other retryable kinds (crash, network, LLM) still retry, inside the same 90 min cap. The user-facing "20 to 60 minutes" copy follows the cap. D-W05 ("a run takes 20-60 minutes") needs the same update when the PO copies the rulings. | W4b |
| Q-E2E-2 | The curator's `chat_with_tools` loop has no real time cap (20 tool rounds x 300 s x retries). Capping it is a change under `curation/`, which is biology scope. | **Leave it for now.** The worker's overall cap bounds it. Recorded as a biology-side follow-up, with no change on the web track. | — |

## Addendum 4, 2026-11-06 (R3c, Stage 1 move)

| ref | question | ruling | relied on by |
|---|---|---|---|
| Q-R3c-1 | `test_streamlit_stage2_orchestration.py::test_out_of_scope_filter_runs_between_stage1_extraction_and_stage2_inference` parses only the app file. It needs one plain-name call each of the Stage 1 extraction, the out-of-scope filter and Stage 2 inference, in line order, with Stage 2 fed the filtered payload. After R3c, the first two live in `service/extraction.py` as call-time `_pipeline.x(...)` calls, so a Q-R0-1 path-only repoint cannot fix it. | **Rewrite that one test.** An edit of about 15 lines beyond Q-R0-1 is authorised for this test only. It must keep guarding the same rule: in `service/extraction.py`, extraction comes before the filter; in the app, `run_stage_one(` comes before Stage 2, and neither Stage 2 nor the first `merge_additions` is fed the raw `stage_one`. The reviewer must confirm that it guards the same rule and that removing the filter or reordering it still makes the test fail. R3d must re-check the test when Stage 2 moves. | R3c, R3d |

## Addendum 5, 2026-11-07 (R3d, Stage 2 move)

| ref | question | ruling | relied on by |
|---|---|---|---|
| Q-R3d-1 | After R3d, Stage 2 no longer lives in the app either, so the Q-R3c-1 order test (`test_out_of_scope_filter_runs_between_stage1_extraction_and_stage2_inference`) fails again. The proposal is a ~20-line rewrite of that one function. It checks the same rule inside `service/extraction.py` (extraction, then the filter, then Stage 2 fed the filtered payload, then the first `merge_additions` seeded from it) and closes R3c review F1. | **Rewrite it again.** This test only. The reviewer must confirm that it guards the same rule, and must verify independently that each of these makes it fail: removing the filter, reordering it, returning the raw extraction, and feeding Stage 2 or the first `merge_additions` the raw payload. | R3d |
| Q-R3d-2 | The headless path has no Streamlit refusal panel for the ambiguous-review gate. In R3d, Stage 1 refuses an ambiguous context before any LLM call, with the same message text (kind `stage_one_failed`). | **Keep what R3d ships, and map it in R5.** In R5, map that refusal to a proper ambiguous-scope error for web users, and move the gate into the service at the same time (one Stage 0 clobber-test edit, authorised then). There is no service copy of the app's `pipeline_error` dict now. | R3d, R5 |
| Q-R3d-3 | The equivalence test missed one R3d mutation: one Stage 2 QA round instead of two gives byte-identical output on the recorded fixture. A 2-line check that every recorded LLM reply is consumed would catch it. | **Add it.** The 2-line edit to the equivalence test is authorised. It is a strengthening, not an edit needed to pass, so merge gate 5's "edited to pass" reject does not apply. The reviewer must confirm that it only adds the check, and that the test still passes 14 on the tip and on base. | R3d, R4-R6 |

Orchestrator note (not a PO ruling): the c060a edit (`tests/test_c060a_requested_pathway_wiring.py`) points its two parse sites at `service/extraction.py` and moves the `OWNED_FLOOR` locator from 1440 to 790 in that file. The orchestrator authorised it under **Q-R0-1** as a path edit, on condition that the reviewer confirms it selects exactly the same two `merge_additions` statements and gate imports as at base.

## Addendum 6, 2026-11-08 (R3 complete; equivalence coverage)

| ref | question | ruling | relied on by |
|---|---|---|---|
| Q-R3d-4 | R3d review F1: the one recorded fixture has no Stage 2 additions, no out-of-scope hits, RAG off and the default export mode, so three mutations pass the equivalence test 14/14. Should a second recording be added? | **Record it alongside R4.** R4 starts now. The second fixture joins the equivalence gate as soon as it works, so R5 and R6 are covered. One heavy job at a time still applies. The Q-R25-2 licence rule applies: CC-BY or CC0 only. | R26, R4-R6 |

Orchestrator note (not a PO ruling): the question put to the PO described the recording as "with RAG on". R2.5 forces `RAG_ENABLED=0` because the cassette does not cover RAG's PubMed and embedding calls, so RAG-on needs harness work, and it could commit retrieved passages whose licences are unknown. Task R26 therefore delivers Stage 2 additions, out-of-scope hits and a non-default export mode first. RAG coverage goes ahead only if it is feasible and every text the fixture stores meets Q-R25-2; otherwise the task stops and reports options and costs to the PO.
