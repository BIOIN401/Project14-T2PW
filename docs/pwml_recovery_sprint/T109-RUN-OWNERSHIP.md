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
| 7 | 2026-09-03T04:29:40Z | alive | **6/20** | `[6/20] PMC12657337 / research -> SCOPE_CONFLICT in 740.4s (12m20s)`. T-108 842.9s. **Organism trap 1 of 3 COMPLETE — both legs aborted at Stage 0, as designed.** The half worth checking rather than assuming is **research mode**, which is the relaxed one: the trap still fires there, same Stage-0 organism contradiction, and **zero `.pwml` files exist under this paper across BOTH modes.** A relaxed export mode does not relax the scope gate |
| 8 | 2026-09-03T04:48:57Z | alive | **7/20** | `[7/20] PMC12421875 / strict -> SCOPE_CONFLICT in 1157.0s (19m16s)`. T-108 842.1s. **Organism trap 2 of 3, gate held, zero `.pwml` under this paper.** **Pace:** 6864 s elapsed for 7 legs = **1.91 h**; naive linear projection **5.45 h** against T-108's measured **6.37 h**. **The projection is deliberately not trusted**: the three organism traps abort at Stage 0 and are cheap, and the expensive legs — `PMC12096016`, `PMC12452463`, `PMC12782028` — are all still ahead. Ample headroom either way against the 18 h internal deadline and the 20 h wrapper |
| 9 | 2026-09-03T04:57:10Z | alive | **8/20** | `[8/20] PMC12421875 / research -> SCOPE_CONFLICT in 492.2s (8m12s)`. T-108 949.3s. **Organism trap 2 of 3 COMPLETE**, both legs aborted at Stage 0, **zero `.pwml` under this paper with both modes now finished** — re-checked after the research leg rather than inferred from the strict-only check at interval 8 |
| 10 | 2026-09-03T05:03:42Z | alive | **9/20** | `[9/20] PMC12312563 / strict -> SCOPE_CONFLICT in 391.7s (6m31s)`. **Organism trap 3 of 3, strict.** All three traps have now aborted at Stage 0 on their strict leg, which is the designed outcome and matches T-107 and T-108. Zero `.pwml` under this paper |
| 11 | 2026-09-03T05:10:23Z | alive | **10/20 — HALFWAY** | `[10/20] PMC12312563 / research -> SCOPE_CONFLICT in 400.6s (6m40s)`. **All three organism traps COMPLETE: 6 of 6 legs aborted at Stage 0, and all three papers carry ZERO `.pwml` files.** Manifest: 10 rows, `scope_conflict 6`, `fail 4`. Elapsed **2.26 h**. **Every remaining leg is a substantive paper** — the two heme papers, the two enterobactin papers including `PMC12096016`, and `PMC12782028`. **The cheap half is behind and the pace projection from interval 8 is now known to be optimistic**, exactly as it was flagged to be |
| 12 | 2026-09-03T05:26:18Z | alive | **11/20** | `[11/20] PMC12856317 / strict -> FAIL (contract) in 955.2s (15m55s)`. T-108 also `FAIL(contract)` here **but on a DIFFERENT GATE**, and the difference is the point. **T-109:** `gate.registry_validation_failed` at `final_pre_export_stage3_gates`, `blocking_issues=1`, `gate_errors=0`. The real reason, dug out of `contract_reports.json` because `RESULT.txt` truncates it to `"Registry validation failed: Registry validation failed:"` — **two ORPHANED REFERENCES**: `/processes/interactions/1/entity_2 unknown entity: HRM3` and `/processes/interactions/2/entity_2 unknown entity: HRM6`. The draw emitted interactions pointing at entities it never declared, and the gate refused. **That is referential integrity, which is Priority 3's subject** — flagged for the scorer, not scored here. **The three release candidates have now produced three different protein sets on this one leg:** T-107 `ALAS1`+`ALAS2` (no ClpXP), T-108 ClpXP present (so the ClpXP identifier gate had something to fire on), T-109 **`ALAS2` alone**. **Three draws, three protein sets, three distinct gate outcomes, all `FAIL(contract)`.** This is what the standing draw-variance trap describes, seen directly rather than argued about |
| 13 | 2026-09-03T05:38:59Z | alive | **12/20** | `[12/20] PMC12856317 / research -> PASS WITH WARNINGS in 760.1s (12m40s)`. **The first PASS of the run.** `stage: research_report`, `blocking_issues=0`, `gate_errors=0`, `review_flags=1`. **No `.pwml` file, and that is correct, not a defect** — research mode produces a research report and a *candidate* pathway for human review, never an importable PWML. **The honest qualifier, recorded now so it is not read off the word PASS later: the pathway is THIN.** `final_mapped.json` carries **1 reaction, 0 transports, 2 interactions, 1 protein, 5 compounds** for a heme-biosynthesis paper. A research-mode PASS with a single reaction clears the gates and is not a demonstration of pathway recovery. **Whether it counts toward research deliverable or research confirmed is the scorer's call and is deliberately not made here.** One detail flagged for scoring: `interactions` moved **1 -> 2** between `stage1_payload.json` and `final_mapped.json`, so something downstream added one |
| 14 | 2026-09-03T05:56:50Z | alive | **13/20** | `[13/20] PMC12180156 / strict -> PASS WITH WARNINGS in 1070.9s (17m50s)`. **T-108 FAILED this leg** (`FAIL(contract)` on `ferrochelatase`, F-147 reproducing). **Read the runner's `PASS` carefully, because it does NOT mean a releasable pathway — and the artifacts say so in four independent ways.** The only export is **`pathway.review_required.pwml` (19563 B)**; there is **NO bare `pathway.pwml`**; the artifacts carry **`review_required: true`** and **`strict_acceptance_eligible: false`**. **And the gold agrees with the refusal:** `PMC12180156`'s `export_rationale` reads *"With zero heme-biosynthesis reactions recoverable, nothing about heme biosynthesis is exportable. At most the paper supports regulatory edges."* The draw extracted **one** reaction, named literally **`"heme biosynthesis reaction"`** — a generic label, not a recovered transformation — plus 1 interaction. **So a bare `pathway.pwml` here would have been an F-100/F-101-class defect, and the gate refused it.** This is the D-089 machinery behaving exactly as ruled: an incomplete pathway **preserved as `review_required` rather than dropped**, and **not** eligible for strict acceptance. **It therefore does NOT move Priority 5**, whose numerator requires `strict_acceptance_eligible`. **Recorded as a gate working, not as a success** |
| 15 | 2026-09-03T06:07:00Z | alive | **14/20** | `[14/20] PMC12180156 / research -> FAIL (no_reactions) in 609.0s (10m09s)`. `operational_failure` and `termination_reason` both **unrecorded** — not a timeout, crash or infrastructure failure. Payload: **reactions 0**, interactions 2, proteins 3, compounds 3. **The counterintuitive part, recorded because it will look like a mode bug to a future reader and is not one:** the STRICT leg on this same paper extracted **1** reaction and the RELAXED research leg extracted **0**. Research mode does not extract more; it *withholds less*. **Both legs are separate draws, and drawing fewer reactions in research than in strict is variance, not a mode inversion.** It also agrees with the gold, which says **zero heme-biosynthesis reactions are recoverable from this paper** — on this leg the pipeline found what the gold says is there |
| 16 | 2026-09-03T06:29:22Z | alive | **15/20** | `[15/20] PMC12096016 / strict -> PASS WITH WARNINGS in 1341.7s (22m21s)`. **The leg the whole D-088 / D-089 / F-173 question is about.** T-108: PASS 1952.9 s, 74367 B. **T-109: `pathway.review_required.pwml` 65100 B, NO bare `pathway.pwml`, `review_required: true`, `strict_acceptance_eligible: false`** — with **`blocking_issues=0`, `gate_errors=0`, `coverage_ratio 0.916667`, `minimum_core_satisfied: True`, 9 core processes accepted, 0 quarantined.** **The INCOMPLETE-CORE CAP held a 65 KB pathway that passed every other gate, on ONE unmatched anchor.** That is F-094's cap doing exactly what D-089 ruled it would keep doing. **BUT the anchor matters, and it REFINES F-173 rather than confirming it.** `unmatched_terms` is **`["EntD (phosphopantetheinyl transferase)"]` and nothing else** — **`ATP`, `NADH` and `Fur` ALL MATCHED on this draw**, where F-169 found all four unmatched on the archived draws. **D-088's expected-consequence table excuses ATP/NADH/Fur and does NOT excuse EntD** — it requires EntD *"remain VISIBLE as an extracted/supporting entity that is not properly wired, unless its required relationship is demonstrated."* **So on THIS draw the cap is held by the one anchor D-088 declines to excuse, and this particular `review_required` is NOT a false negative in D-088's terms.** F-173's general claim — that the instrument *cannot distinguish* the two legs — still stands; what this draw shows is that the limitation did not bite here, because the draw happened to match the three cofactor-class anchors. **Recorded as a refinement, measured on one draw, and not as a retraction** |
| 17 | 2026-09-03T06:50:43Z | alive | **16/20** | `[16/20] PMC12096016 / research -> PASS WITH WARNINGS in 1281.2s (21m21s)`. `stage: research_report`, `blocking_issues=0`, `gate_errors=0`, **`review_flags=0`** — the only clean-flag leg of the run so far. No `.pwml`, correct for research mode. `final_mapped.json`: **3 reactions, 5 interactions, 6 proteins, 7 compounds**, and the reactions are **named for their enzymes** — `EntC reaction`, `EntB isochorismatase reaction`, `EntA reaction` — rather than the generic `"heme biosynthesis reaction"` label that leg 13 produced. **This is a substantively richer draw than the heme legs, on the paper the strict leg also recovered 9 core processes from.** Still not scored here: whether 3 reactions constitutes research-confirmed is the scorer's judgement and the enterobactin pathway has more steps than three |
| 18 | 2026-09-03T07:05:40Z | alive | **17/20** | `[17/20] PMC12452463 / strict -> FAIL (contract) in 896.5s (14m56s)`. **8 blocking issues**, `gate_errors=2`, at `final_pre_export_stage3_gates`. Codes: `actor_schema_not_canonical`, `gate.expected_plus_tokens_found`, `gate.generated_protein_complex_wrapper_enterobactin_synthase_complex_`, `gate.composite_validation_failed`. **The blocking-issue count on this leg across four release candidates is now 7 -> 3 -> 6 -> 8** (T-106, T-107, T-108, T-109). **T-108's log already retired the earlier "improved 7 to 3" claim as draw variance; a fourth value in the same band settles it beyond argument.** The count is a draw property, not a trend, and no wave should ever again be credited or blamed for moving it. **`gate.generated_protein_complex_wrapper_...` is the F-147-adjacent wrapper seam and remains UNCHARTERED by standing instruction — observed, not touched** |
| 19 | 2026-09-03T07:22:50Z | alive | **18/20** | `[18/20] PMC12452463 / research -> PASS WITH WARNINGS in 1028.6s (17m08s)`. Research report, no `.pwml`, as correct for the mode. **Same paper, opposite outcome by mode: strict FAILED on 8 blocking issues while research passed** — which is the modes behaving as designed, not a contradiction. Research annotates and flags; strict refuses. The two legs are separate draws over the same paper and the strict refusal is the one that carries the export decision |
| 20 | 2026-09-03T07:37:19Z | alive | **19/20** | `[19/20] PMC12782028 / strict -> PASS WITH WARNINGS in 868.4s (14m28s)`. **THE CONTROL LEG, and D-088's required consequence HOLDS with the artifacts naming the exact enzymes.** `pathway.review_required.pwml` **35295 B — byte-identical in size to T-107's 35295 B** — **NO bare `pathway.pwml`**, `review_required: true`, `strict_acceptance_eligible: false`. `coverage_ratio` **0.571429**, `core_accepted` 3. **`unmatched_terms` = `["oxysterol", "MSMO1", "SQLE", "FDFT1", "HMGCR", "HMGCS1"]`.** **`HMGCR` and `HMGCS1` ARE the mevalonate arm** — HMG-CoA reductase and HMG-CoA synthase — with `FDFT1` and `SQLE` the steps immediately downstream and `MSMO1` the methylsterol demethylation stage F-168 named. **D-088 requires this leg to "remain incomplete -- its upstream mevalonate reaction arm is genuinely absent," and it does, by name.** **BOTH required consequences now hold SIMULTANEOUSLY on one fresh draw: `PMC12096016/strict` held by `EntD` alone (the anchor D-088 declines to excuse) and `PMC12782028/strict` held by the genuinely missing arm. A change clearing both would have been a reject; nothing cleared either.** **And Candidate A is re-refuted on fresh data:** `min_core_coverage` is 0.5 and this leg reads **0.571 >= 0.5** with `minimum_core_satisfied: True`, so relying on the existing thresholds would have RELEASED the leg whose mevalonate arm is missing — exactly what `DECISION-PACKET-D088-RUNTIME-CAP.md` § 2 measured at 0.538 on the archived draw |
| 21 | 2026-09-03T07:51:36Z | **EXITED** | **20/20** | `[20/20] PMC12782028 / research -> PASS WITH WARNINGS in 857.0s`. `FAILURES: 13 of 20`. Wrapper exit **1** (`nonzero`) — the expected code when not every leg passes, **not** an infrastructure failure; T-107 and T-108 exited 1 likewise. **17824.00 s = 4.95 h** of a 72000 s ceiling. **`FINAL SURVIVING COUNT : 0`**, **`cleanup : success`**, 43 descendants observed and 43 terminated, heavy lock **acquired and released**, 4 pre-existing reported and never killed |

## 5. Closure — CLOSED 2026-09-03T08:05Z

**T-109 ran exactly once and is complete. Verdict `NOT ACCEPTED` — `T109-RESULT.md`.**

| Closure check | Result |
|---|---|
| Wrapper exit | **1** (`nonzero`) — expected when not every leg passes. **Not** an infrastructure failure |
| Duration | **17824.00 s = 4.95 h** of a 72000 s ceiling (24.8%) |
| Timeouts | **0** — the first release candidate in the sprint with none |
| `FINAL SURVIVING COUNT` | **0** |
| `cleanup` | **success** |
| survivors | `[]` |
| Heavy lock | `acquired=true released=true`; `C:/t/heavylock` **absent** after |
| Sprint-owned Python after | **zero** — verified by **full command line** |
| G11 | `check --task T-109` — see the commit that files this |
| Gold | `36f4b7b6…` unchanged before and after |
| `acceptance.py` | `4bd893ac…` unchanged |
| `streamlit_app.py` | `47e4fafa…` unchanged, still uncommitted |
| `main` | `7531692` — never written |

**One honest deviation from the process baseline, recorded rather than smoothed.** The two
`ms-python.isort` language servers have **different PIDs after the run** — `177556`/`271968` and
`177596`/`272160`. The IDE respawned them mid-run. The **count is unchanged at two** and the command
line is byte-identical, and **that is exactly why the command line is the identity and the PID never
is.** Neither was ever a cleanup target.

**Ownership held end to end by `project14-t2pw-51` `[e2c249]`. No transfer occurred, none was needed,
and no job was ever unowned.** Monitor task `b62ia9a1u` stopped at completion; wrapper task
`bjyoa5sl4` exited on its own.

**Status: CLOSED.**
