# T-109 — run-ownership record

**The single authoritative record of who owns the T-109 milestone run.** Opened before the run was
staged; updated at each monitoring interval; closed only after the wrapper exits, cleanup is
verified and the result is committed.

**An unacknowledged background process is a G11 violation.** This file exists so that no state of
this run is ever unowned. It follows `T108-RUN-OWNERSHIP.md`, which recorded the pattern that
worked for a 6.37 h run held end to end by a single session.

---

## 1. Owner

| Field | Value |
|---|---|
| Session identity | `project14-t2pw-51` `[e2c249]` |
| Role | Lead Orchestrator · Integration Authority · **T-109 Run Owner** |
| Repository | `C:\Users\Angad\Desktop\SummerBIOIN\Project14-T2PW` (**primary checkout**, not a worktree) |
| Integration branch | `sprint/pwml-recovery` |
| `main` | local `7531692` / remote `03f1af5` — **neither ref written, before or after** |

## 2. Exclusivity — how it was established

`ListAgents` returned **26 peers. Not one is live in this repository.** The only peer that has ever
held a stand-down here, `Assess RAG pathway mining infrastructure` `[2bdab1]` — the session earlier
handoffs call `project14-t2pw-93` — is **Remote Control, offline.** Every other peer is a cloud or
Remote-Control session on an unrelated project (StockScreener, LoadLink, quant work), idle or
offline.

**Independently corroborated at the machine, which is the check that actually binds:**

- `C:/t/heavylock` **absent** before the claim.
- The full process table shows **exactly two** `python.exe`, both
  `ms-python.isort-2026.6.0/bundled/tool/lsp_server.py`, **matched on command line, never on count
  or PID** — the count has changed twice in this sprint and is not an identity.

**Transfer recipients: none.** No live peer is authorized for sprint work, so **no transfer is
possible** and none is planned. This session holds the run to completion. If an unavoidable session
failure occurs, a transfer is valid **only** to an already-live named peer that explicitly
acknowledges the wrapper PID, task id, output path, heavy-lock state, last observed progress, and
the responsibility to monitor through cleanup. **A transfer nobody has accepted is not a transfer.**


## 3. The run

| Field | Value |
|---|---|
| Milestone identity | **T-109** — a NEW milestone. **Not** a re-run, re-score or re-reading of T-107 or T-108, both of which stay immutable and `NOT ACCEPTED` |
| Authority to launch | product-owner ruling of 2026-09-02, recorded as **`D-089`**; readiness in **`T109-READINESS.md`**, decision **GO**, 19 of 20 rows green and row 14 proved in this directory |
| Branch tip at claim | **`a844443f75bf04424d2faf482aae53bc099df920`** |
| `local = origin/ = git ls-remote` | **verified equal at `a844443f`** before the claim |
| Run directory | **`runs_verify/2026-09-02_2052`** (staged, verified, to be continued **without `--fresh`**) |
| Wrapper | `docs/pwml_recovery_sprint/evidence/bounded_run.py`, build `sha256:83d13954…` (unmodified, identical to T-108's) |
| Basetemp / log root | `C:/t/x109/` (parent pre-created) |
| Wrapper stdout/stderr log | `C:/t/x109/log/t109_run.log` |
| Wrapper task id | **`bjyoa5sl4`** (Bash tracked background; notifies this session on exit) |
| Monitor task id | **`b62ia9a1u`** (persistent; one event per leg completion and per failure signature) |
| Process chain | outer `bounded_run.py` **191228** -> job root **206276** (the PID named in the lock) -> `batch_run.py` **206044** -> **203820** -> leg children |
| Per-leg ceiling | **3600 s default, NO override** — `leg_timeout_overridden: false`, verified **in the staged directory before launch** |
| Wrapper hard ceiling | **72000 s (20 h)** — same as T-108, for comparability |
| Internal `--deadline` | **18 h** — same as T-108 |
| Expected duration | **~6.4 h**, from T-108's measured 22929.17 s on the same corpus at the same ceiling |

### 3.1 Pre-launch gates, every one measured today and none carried forward

| Gate | Result | Evidence |
|---|---|---|
| Provider and nine pinned models | **PASS** — `openrouter`, `deepseek/deepseek-v4-flash` ×9, temp 0, retries 3, key present (73 ch, `sk-or-v1-`), **never printed** | G11 `T-109/01` |
| Stage-only preflight | exit 0, survivors 0 | G11 `T-109/02` |
| `--verify-plan` | **`verdict: OK`**, 10 cases, **search calls: 0**, all 10 `[pinned_override]` | G11 `T-109/03` |
| Staged-directory verification | **`T108_STAGE_VERIFY: OK`** — `find_resumable` MATCH, 20 pairs / **20 pending / 0 legs / 0 RESULT.txt**, gold blob `36f4b7b6…` identical in working tree, HEAD and expected, ceiling 3600 s `overridden: False` | G11 `T-109/04` |
| SMOKE (merge gate 10) | **503 passed, exit 0**, pin verdict `refused=false, violations=[], foreign_src=[]` | G11 `ORCH-720/01` |
| gold-readers | **456 / 0 / 8 / 0**, exit 0 | G11 `ORCH-720/02` |
| battery + F-146 | **`battery=0/29  F146=REJECTED  C1..C6=0`** | G11 `ORCH-720/03` |
| mutation harness | **17 mutations, SURVIVORS 0** | G11 `ORCH-720/05` |

**One red result deliberately carried into the launch rather than hidden:** `chunk_d_gate.py` is RED
in the primary checkout (`run-core 159/160`, `node15 0/1`). It is **F-174**, it is **not** a readiness
row (`TEST_MATRIX:244` excludes Chunk D from the smoke gate), **it cannot be a code regression** — the
only commit since the last green Chunk D touched three evidence artifacts and no `src/`, `tests/` or
`scripts/` — and **T-108 ran in this same checkout**, so it does not make the two runs less
comparable. Node 2's precise lever is registered **OPEN**.

## 4. Monitoring log

Recorded at every interval: wrapper alive · current paper/mode · completed legs · output growth ·
heavy-lock holder · last log mtime · provider errors · retry and timeout telemetry · finalization
reserve · partial-payload preservation.

| # | UTC | Wrapper | Legs | Notes |
|---|---|---|---|---|
| 0 | 2026-09-03T02:5x Z | not launched | — | ownership claimed; all pre-launch gates green; this file committed and pushed **before** the wrapper started |
| 1 | 2026-09-03T02:54:33Z | **LAUNCHED** | 0/20 | Lock acquired, token `T-109:206276:e322f748e2aaee66`. `CONTINUING the incomplete run 2026-09-02_2052 (no --fresh given)` — `already recorded : 0`, `still to do : 20`. `whole-night deadline : 18.0h from now`. Leg `[1/20] PMC12444477 / strict -> starting (timeout 3600s)` |
| 2 | 2026-09-03T03:34:09Z | alive | **1/20** | `[1/20] PMC12444477 / strict -> FAIL (contract) in 2373.8s (39m33s)`. **This leg FINISHED where both prior release candidates lost it to the clock** — T-107 `TIMEOUT 1798.3s`, T-108 `TIMEOUT 3600.8s`, here **2373.8 s**, and 16 artifact files were produced where T-108 preserved only C-111's terminal trio. **The failure is a LIFECYCLE one, not biology:** `post-pipeline validation failed: 1 blocking issue(s) at final_pre_export_gate_lifecycle [final_gate_report_missing]`, issue code `gate.gate_lifecycle`, `blocking_issues=1`, `gate_errors=1`. `operational_failure` and `termination_reason` are both **unrecorded**, so this is not a timeout, crash or infrastructure failure. Four further runtime-schema findings are explicitly **informational, not blocking**. **Consequence to measure at scoring, not asserted here: `LpxH` may become measurable on T-109**, which `T108-READINESS.md` § 5 flagged as unverified because both `PMC12444477` legs had no payload to inspect. **Not claimed yet, and one leg is not a trend** — the standing trap binds in both directions |
| 3 | 2026-09-03T03:36:39Z | alive | **2/20** | `[2/20] PMC12444477 / research -> FAIL (unknown) in 149.1s (2m29s)`. T-108: `PASS WITH WARNINGS 2446.5s`. **Checked for infrastructure degradation because 149 s is fast enough to look like one, and it is NOT:** `Extraction failed: Chunk 2 failed to produce valid JSON` at **stage1**; `operational_failure` and `termination_reason` both **unrecorded**; Stage 0 `status=ok finish_reason=stop attempts=1`; all three Stage-1 extraction calls recorded `finish_reason=stop`. **`stop`, not a truncation and not a transport error** — the model completed normally and emitted text that was not JSON. No HTTP error, no rate limit, no retry exhaustion. **`LLM_MAX_RETRIES=3` did not fire, because a JSON-validity failure is not a transport failure** — worth knowing before anyone reads the retry budget as covering this. **This is a draw failure and the pipeline classified it correctly.** Legs 1 and 2 have now moved in OPPOSITE directions against T-108, which is what draw variance looks like and is not evidence about the provider. **Threshold recorded IN ADVANCE so it cannot be rationalised later: three consecutive fast stage-1 JSON failures would make this an infrastructure degradation rather than a result, and it would be REPORTED as one — not silently completed and not aborted, because T-109 is one-shot and aborting burns the identity** |
| 4 | 2026-09-03T03:50:15Z | alive | **3/20** | `[3/20] PMC13231680 / strict -> FAIL (no_reactions) in 816.0s (13m35s)`. **Negative control, and it behaves as one.** T-108 `FAIL(no_reactions) 624.8s`; T-107 `FAIL 679.9s`, triaged CORRECT. **Q1 ruling condition 3 verified from the artifacts, not assumed:** `operational_failure` **not recorded**, `termination_reason` **not recorded**, so the empty result is not caused by timeout, crash or infrastructure failure. Payload confirms **reactions 0, transports 0**. **The runner's `FAIL` token is the known-misleading one** — `runner.py:717` has no gold access; `PASS_NEGATIVE_CONTROL` is a **scorer** verdict and is verified at scoring, never asserted here. **Also ends the degradation watch opened at interval 3:** this is a different failure kind, at a different stage, taking 5.5x longer, so the three-consecutive-fast-JSON threshold is not approached |
| 5 | 2026-09-03T03:59:17Z | alive | **4/20** | `[4/20] PMC13231680 / research -> FAIL (no_reactions) in 541.3s (9m01s)`. **Matches T-108** (`FAIL(no_reactions) 360.4s`); **T-107 had `PASS 795.2s`** on this leg. `operational_failure` and `termination_reason` both **unrecorded**; payload **reactions 0, transports 0**; entities species 1 / compounds 2 / proteins 1 (T-108: 1 / 3 / 2). Stage 0 `ok / stop / attempts 1`. **What this is NOT: evidence that F-146 is fixed.** F-146 is this leg retaining an invented reaction and it was Priority 2's single row on T-107. Its absence on a second draw is still not a fix — the standing trap binds in both directions, and T-108's own log said so about the first. **NEW, and it corrects what I wrote at interval 3:** the RAG prose extraction here shows `finish_reason: 'length'` on **attempts 1, 2 AND 3** — so **`LLM_MAX_RETRIES=3` DOES fire, on TRUNCATION**. Leg 2's JSON-validity failure never retried. **The retry budget covers `length`, not malformed content**, and the two failure classes are now measured side by side rather than inferred. **Whether the RAG prose token ceiling is systematically too low is a question for scoring, not a claim here** |
| 6 | 2026-09-03T04:17:19Z | alive | **5/20** | `[5/20] PMC12657337 / strict -> SCOPE_CONFLICT in 1081.5s (18m01s)`. **Organism trap 1 of 3, and the gate held.** T-108 1011.2s, T-107 547.9s — slower each time, same outcome. `Stage 0 read organism 'Escherichia coli' but the batch requested 'Bacillus subtilis'`, `eligibility_stage0_conflict_aborts=True`. **The scored property verified rather than assumed: ZERO `.pwml` files exist anywhere under this paper**, so nothing was exported carrying the requested-but-wrong organism label. That is what `PMC12657337`'s gold `export_rationale` scores |
