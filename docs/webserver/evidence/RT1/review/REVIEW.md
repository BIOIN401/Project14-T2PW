TASK: RT1  BRANCH: ws/RT1-llm-timing  REVIEWED HEAD: 9afa3c14
VERDICT: APPROVE WITH NOTES
GATES: boundary ✓ · no-biology ✓ · behaviour-preserved n.a. (not an R-task; non-web path unchanged by reading) · tests ✓ · evidence ✓ · secrets ✓ · result-doc ✓

No blocking defect found. Nothing was run, per your instruction. No live OpenRouter call exists, so real response field shapes and real abandonment behaviour are not established.

FINDINGS (all notes), paths under C:/t/ws-RT1
F1 [note] src/t2pw/llm/client.py:340,388. The cap cannot distinguish a hung request from a legitimately long one. Margin over the 718 s longest good call is 1.8 s, from 38 legs. If generation time scales with paper size (not established), a paper slightly bigger than the sample hits the cap on every attempt: 8 x 12 min, parent kill at 90 min, `timeout`, where the old path would have finished at about 13 min. RESULT Q-RT1-A presents the risk only as a provider outage, so it understates this. The number is PO-ruled; read `call_cap` rows in the first real jobs.
F2 [note] src/t2pw/service/postpipeline.py:1417 with src/t2pw/stoich/agent.py:479,555,583. When `use_stoich_agent` is on, these call `_client` directly and never appear in the trace, so "per-LLM-call logging" has a gap. Fixing it is outside the allowed files. RESULT does not mention it.
F3 [note] tests/test_llm_call_cap.py:283 is tautological (literal vs literal). Line 190 compares the constant to a copy of itself. No test ties the four strings to real call sites, so a renamed label would silently switch the cap off. I verified them by reading (Q8).
F4 [note] src/t2pw/llm/client.py:918-922. On ok and empty attempts `duration_s` includes the 0.35 s spacing sleep, because `note` runs after it. Immaterial, but worth knowing when reading traces.
F5 [note] src/t2pw/llm/client.py:370 and RESULT:47. "Ends on the SDK's own timeouts" conflicts with the comment at client.py:~336 (keep-alive defeats them). An abandoned request can live as long as the provider streams. RESULT also says "3 stages"; there are 4. Abandoned generations may also count against provider concurrency, which belongs under Q-RT1-B.
F6 [note] src/t2pw/llm/client.py:1070. "on all N attempts" is true only of the last error when attempts were mixed. The pre-existing timeout path has the same flaw.

QUESTIONS 1-8
1. Non-web: `_create` is the plain call on the caller's thread, adding one stack frame. `begin_attempt`/`observe` are gated no-ops. `_publish_attempt` takes the old branch. The new dataclass fields are init=False, compare=False, and no `asdict` touches them. Same kwargs, exceptions and rows.
2. Scope: the worker reaches `service.pathway:run_pathway` (worker.py:70), the only setter. Batch `run_one_pathway` calls `run_paper` (driver.py). The optional `driver_bridge` run-fn is uncapped. `finally` restores the flag on every in-process exit; a kill ends the process.
3. Status: no src reader branches on attempt-row status. `attempt_log` only passes through `_bounded_attempt`, which has no vocabulary check. `response_status` can be `call_cap` only on the raising path, and nobody reads `exc.diagnostics`. All-capped gives `LLMOperationTimeout`, identical to the old path. Classification still matches "llm"/"timeout"; the new text hits no network or contract marker and not `BUDGET_SPENT_MARKER`.
4. Deadlines: the cap is not shrunk by the job deadline, but neither was the old wait. The parent kills via Job Object or killpg at the deadline regardless of `Event.wait`, so no new overrun past 90 min. A kill mid-wait skips the in-process timeout path, also as before. `worst_case_call_seconds` still prices 300 s per attempt, understated as before.
5. Threads: the worker only runs `create` and fills a local box, with one branch and no streaming or fallback. `_record_usage` and `observe` are caller-only, so the late-write claim is true. SDK retries are 0, so no re-send. The shallow copy is sufficient: only top-level `response_format` is mutated, and `messages` is not. A provider error landing exactly at 720 s reads as `call_cap`, which is harmless. httpx thread safety is from library documentation, not tested.
6. Context: client.py reads no ContextVar or thread-local; `current_diagnostics` and the ladder deadline are read on the caller's thread. Replay patches `create` on the `_client` instance behind a lock, so `run_pathway` arms really cross the worker thread. That covers the happy path only.
7. Trace: not reachable. `resolve_download` is by key only, there is one FileResponse, static is STATIC_DIR only, there is no zip or directory listing in web/service, and the curator queue copies named keys. It is written directly to the attempt's `out/` and survives. Content is hashes, counts and clipped, credential-redacted errors; `reason` may hold a clipped provider error, in a private file. Activate, write and fsync are all guarded, so a full disk or bad path cannot break a job. A pre-existing trace is left alone. `app/reaction_evidence.py` expects `paper_id` on the first line, but web/service do not import it.
8. Allowlist: exact matches. preprocessor.py:165; pipeline.py label map into `_issue` at :3855, used for all Stage 1/2 draws including ladder rungs; rag/extract.py:144 via `chat`. Localized repairs use other labels and are uncapped.

Q9: new tests are otherwise genuine; 15 reverts fail as claimed, and the surviving first M12 is disclosed. 720 > 718 holds (see F1).
Q10: no scope creep or dead code. A worker thread is the simplest non-streaming mechanism.

Evidence: all 13 run reports show 0 survivors; every pass and revert count matches the outputs; no src/tests change after the tested tip 07ee8421. Only the two new tests were added; no existing test edited. No .env, cache or quarantine-named file.

OPEN QUESTIONS
Q-RT1-B: orchestrator may settle by checking `generation_id` cost after the first real job; escalate only if material.
Q-RT1-C: orchestrator (the one-job-per-process rule stands).
Q-RT1-D: orchestrator; keep `call_cap`, consumers verified as opaque. It does appear in public-downloadable-class boundary reports only if those keys are in `report_paths`; the text is harmless.
Q-RT1-A: product owner, reframed with F1: the risk includes long legitimate papers failing where they used to succeed, and the standing no-regression rule applies.
