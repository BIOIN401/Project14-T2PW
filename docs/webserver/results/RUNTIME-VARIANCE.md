# ANSWER FIRST
No single stage explains 35 to 78 min. Each web run lost its extra time in a different stage, next to one logged failed LLM call (L). The cause is a heavy tail of single slow or empty LLM calls, not more calls and not bigger payloads. The R5 code is not implicated on the evidence (L), but one post-R5 run cannot fully separate it.

# 1 and 2. Per stage, three web runs (minutes, from intermediate-file mtimes, M)
| Stage | E2E-2 | E2E-3 | E2E-4 |
|---|---|---|---|
| Stage 0 + Stage 1 (3 draws, serial) + merge | 6.2 | 5.4 | **57.6** |
| Stage 2 (2 rounds) | 4.9 | **24.3** | 1.0 |
| RAG round 0 + first ID map | 1.2 | 4.2 | 3.0 |
| Normalization + DB mapping | 0.95 | **8.7** | 3.7 |
| Audit (1 round, 3 candidates) + gap | 6.3 | 6.2 | 5.7 |
| Curator + remap | **15.8** | 2.8 | 6.7 |
| Enrichment + export + PNG | 0.1 | 0.2 | 0.2 |
| Total | 35.4 | 51.7 | 77.9 |
Run dirs: E2E-2 = C:/t/ws-e2e + C:/t/e2e2 (driver bridge, 5 Nov 20:51); E2E-3 = C:/t/ws-E2E3c + C:/t/e2e3c-run2 (driver bridge, 9 Nov 11:54); E2E-4 = C:/t/ws-E2E4 + C:/t/e2e4-run (service run_pathway, 9 Nov 15:11). e2e3c-run/work holds only request.json (nothing). The E2E-2 stage split agrees with the table already in docs/webserver/results/E2E-2.md.
Retries (M, extraction_boundary_report.json): E2E-2 none; E2E-3 Stage 2 round 2 attempt 1 empty finish_reason=length; E2E-4 Stage 1 draw 3 attempt 1 empty finish_reason=error. No failover visible (no provider routing in client.py).
Countable calls are identical in all three (Stage 0 x1, 3 draws, 2 Stage-2 rounds, 3 audit candidates). Curator, gap resolver, alias lookup and RAG call counts are unknown.
Payloads are NOT larger in the slow run: stage1_payload 32.9/34.3/31.9 KB, merged 72.6/75.6/71.1 KB, final_mapped 203/213/163 KB. E2E-4 is the smallest.
Does not exist for any web run (M): per-call latency, tokens in/out, provider/failover, curator/gap/alias/RAG call counts. child.log retry lines have no timestamps on the service path.

# 3. Batch legs (runs_papers; 66 leg traces; 38 post-budget-fix legs, 1,397 calls)
- Same paper (PMC7059786, 36,169 chars; web reads 35,999) ran 7 times in batch with per-call traces: 21.4, 21.7, 27.8, 36.0, 47.2, 48.9, 73.6 min. The three clean legs are 21 to 28 min. Excess in the others sits in failed or slow calls: ORCH-767 two empty Stage-2 calls (17.5 + 12.8 min); ORCH-784 two empty finish_reason=error audit calls (25.0 + 21.9 min); ORCH-785 and ORCH-796 one empty Stage-1 call (14.4, 11.1 min); ORCH-781 one good 9.5-min audit call. (M) The web runs (35, 52, 78) sit inside the batch range 21 to 74, E2E-4 about 4 min above.
- Post-fix, all 38 legs: failed calls are 21 of 1,397 (1.5 percent) but hold 15 percent of traced time (322 of 2,158 min); 11 of 38 legs lost 10 min or more to them. Failed call length: Stage 1 median 8.1 (max 14.5), Stage 2 median 16.4 (max 24.8), audit median 25 (max 25.8), against good-call medians of 1.7 to 2.4 min. (M)
- Paper size tracks time across papers (Spearman 0.61 over 39 strict legs, 36k to 201k chars; 18 of 41 strict legs exceeded 60 min, max 180). Reactions do not (-0.05). Within one paper the spread is up to 3.4x. Time of day: weak (rho 0.23, n=33, confounded schedule): G at best. Provider: not recorded. (M for the numbers)
- 37 trace gaps over 300 s and zero attempts with status "timeout" in 1,073 attempts: the 300 s setting (client.py:34, 55-59) does not bound call wall time. (M)

# 4. Recent code (R5)
Same LLM-calling functions on both paths; R5 only replaced the AppTest driver (service/pathway.py:213-440). End-of-run additions (PNG, result.json) were written within 0.02 s of the PWML in E2E-4 (M). The excess is in Stage 0 + 1, which R5 did not touch, and pre-R5 batch runs reach 74 min on the same paper (L: provider call-time tail, not R5). Cannot separate with one post-R5 run and no per-call data.
Caveat: E2E-4's ~51 excess minutes exceeds any single failed call seen (max 28 min), so it was either an unusually long call or several slow ones. Unknown.

# 5. Options to shorten, ranked (none implemented)
1. Wall-clock cap per LLM call, then the existing retry. client.py:721 and :947 (timeout kwarg), retry loops :745 and :960. Needs streaming or a watchdog thread. From 38 post-fix legs: a 12-min cap on failed calls saves 113 min total (3.0 min per leg average, 10 to 40 in legs that hit a failure); 10-min cap saves 139. At 12 min it would cut no good Stage 1, Stage 2, preprocessor or RAG call (longest good 11.97, 6.94, 3.49, 7.65) but would cut 3 good audit calls (12.7, 15.0, 32.6), so exclude audit and curator. Class: no biology effect for calls that were failing; the number is a PO decision; a cap below the longest good call is deferred (no-loss not shown).
2. Run the 3 Stage-1 draws concurrently. stage_one_ensemble.py:1144 (for index in range(1, n+1)); draw body pipeline.py:275-292. About 3 to 4 min per run when nothing fails; does not shorten one long failed call. No biology effect expected (same request, merge keyed by draw index). Verify first: current_diagnostics() and deadline hooks across threads, provider tolerance for 3 parallel requests.
3. Concurrent audit candidates, postpipeline.py:982. About 3.5 to 4 min. DEFERRED, changes results: the serial loop drops candidates 2 and 3 after 240 s (postpipeline.py:975-977), so running all three changes the selected candidate. PO decision.
4. Curator limits (pathway_curator.py:316-326, max_tool_rounds=20 at :325; called at postpipeline.py:1370). Possibly 5 to 12 min in a bad run; unknowable without call data. Capping rounds changes output: deferred. Only option 1 applied to chat_with_tools is neutral.
5. Faster provider route or lower reasoning effort (none set; client.py request kwargs :716-722, :942-948). Savings unknown (provider not recorded). Provider route needs an A/B (quantization can differ); lower reasoning effort changes extraction: deferred, PO decision.
6. Alias lookups in mapping, serial one per unresolved protein, 128k default (map_ids.py:445, :5439). Possibly the 8.7 min in E2E-3 (G). Concurrency is neutral if cache writes stay ordered; a small token cap is not (ORCH-755 showed small budgets give empty answers). Log first.
Not options: raising RUN_TIMEOUT_S (web/retry.py:42, 90 min) only avoids false failures. Large papers (200k chars, 106 to 180 min in batch) would exceed 90 min.
Suggested order: logging (below), then option 1, then option 2.

# 6. Missing logging that would have answered this
1. Activate the leg trace in the web path: client.py:304-345 (_publish_attempt) is a no-op unless a batch leg is active; only batch/runner.py:1151 calls leg_trace.activate. Calling it at the start of service/pathway.run_pathway gives per-call stage, status, finish_reason, chars, hashes and stage_begin/end.
2. Add started_at and duration_s per attempt in CompletionDiagnostics.note (client.py:386-412); today latency is a delta guess.
3. Store per-call prompt, completion and reasoning tokens and the OpenRouter provider/generation id (client.py:118-124 keeps only in-process totals).
4. chat_with_tools (client.py:900-1088) never calls diagnostics.note, so the curator (widest spread) is invisible; its report is deleted with the post-pipeline dir and its exceptions are swallowed (postpipeline.py:1383-1384).
5. Give the alias chat() at map_ids.py:445 a stage_name (it shows as "chat").
6. Timestamp child.log on the service path; write per-stage wall times into run_outcome.json.

# 7. Could not be determined
- Which call(s) produced E2E-4's ~51 excess minutes in Stage 1.
- What curator, gap resolver, alias lookups and RAG did in any web run (no call data).
- Provider, failover and token counts for any web run.
- Whether time of day or provider load matters.
- Whether R5 changed anything beyond end-of-run rendering (no positive evidence it did).
- E2E-2 ran four days before E2E-3/4; I did not diff its .env budgets.
