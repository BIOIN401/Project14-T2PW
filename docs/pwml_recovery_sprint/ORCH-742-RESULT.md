# ORCH-742 — three parallel workstreams: result and integration proposal

2026-09-23 · Lead Orchestrator · **Nothing has been merged into `sprint/pwml-recovery` or `main`.**
Nothing was pushed. Production stays frozen at `5f6a7aa3`.

## Status

| Worker | Branch | Status | Tests (lane tip, independently re-run) | Commit | Blockers | Ready to integrate? |
|---|---|---|---|---|---|---|
| **A — pathway mining** (`ORCH-742a`) | `experiment/pathway-mining` | fixes approved (`APPROVE_WITH_FINDINGS`, 1 round). **Pilot ran: BLOCKED for evaluation** | mining 226 passed (was 222) · SMOKE 508 feature-off · G9: 3 tests fail at `9c085967`, pass at tip | `23e1ae83` | merging brings the **whole unreviewed mining package** (see §4); pilot findings F1–F3 | **No.** Keep it on the experiment branch |
| **B — reaction evidence UI** (`ORCH-742b`) | `feature/reaction-evidence-ui` | approved (`APPROVE_WITH_FINDINGS`, after 2 repair rounds) | model 19 · view 9 · app hook 1 · SMOKE 508 · Chunk D core 160 | `e328961e` | needs a **product-owner D-xxx** authorizing the 8-line hook in frozen/PROTECTED `streamlit_app.py` | **Yes, once that decision exists** |
| **C — evaluation v2** (`ORCH-742c`) | `eval/reaction-rag-metrics` | approved (`APPROVE_WITH_FINDINGS`) + bio `SOUND_WITH_FINDINGS`, after 2 repair rounds | eval_v2 60 passed + 1 strict xfail · SMOKE 508 · gold-readers 465/8, the same as base | `a51e9609` | none blocking. One gold flag (PMC12657337) is not certified by the bio-auditor | **Yes** (lowest risk: new files only, no production file touched) |

Every run in this record went through `bounded_run.py`, and **every job ended with FINAL SURVIVING COUNT 0**.

## 1. What each branch changed

| Branch | Non-evidence files | Evidence / record files |
|---|---|---|
| `experiment/pathway-mining` `9c085967..23e1ae83` (the approved lane delta) | `src/t2pw/mining/{preflight,budget,controller}.py`, `tests/test_mining_{stability,budget}.py`, `docs/pathway_mining/{known_limitations,limits_and_stopping,testing}.md`: 8 files, +159/−7 | 13 under `docs/pathway_mining/evidence/` |
| the same branch measured from the sprint base, `9f13188e..23e1ae83` (**what a merge actually brings**) | the entire mining package (24 modules plus UI), 12 test files, `docs/pathway_mining/*`, and the 17-line hook in `streamlit_app.py`: 60 files, +16,885 | 21 |
| `feature/reaction-evidence-ui` | new `src/t2pw/app/reaction_evidence.py`, `reaction_evidence_view.py` · `streamlit_app.py` **+8 lines at EOF, 0 changed** · 3 test files · `docs/reaction_evidence_sample.txt`: 7 files, +1,531 | 106 (G11 reports and pins, 4 trimmed fixtures) |
| `eval/reaction-rag-metrics` | new `src/t2pw/bench/eval_v2/` (8 modules + `gold_v2_supported_reactions.json`) · 4 test files · `docs/evaluation_v2/`: 18 files | 45 |
| `experiment/pathway-mining-pilot-001` `23e1ae83..dcb64fc4` | none. It is the pilot record only, under `docs/pathway_mining/pilot/001/` | 70 files, 2.6 MB |

## 2. Overlap and conflicts

- **One shared file:** `src/t2pw/app/streamlit_app.py`. Mining adds 17 lines after `st.set_page_config` (candidate lines 142–157). The UI adds 8 lines at EOF (7254–7261). **The merge is textually clean** (`git merge-tree`, then real merges). The runtime interaction was **tested, not assumed**. With `T2PW_PATHWAY_MINING_ENABLED=true` on the real app (a new, labelled AppTest probe), PWML mode shows the Reaction Evidence section with no exception, and mining mode omits it with no exception. Widget and session keys do not collide.
- **Eval** shares no file with either lane.
- **Your protected, uncommitted `streamlit_app.py` edit** still applies to every candidate. A read-only copy is `C:/t/protected/user_streamlit.patch`; it lands at offset +17 on the full candidate. The protected fingerprint must move after integration: `47e4fafa…` becomes **`20d31a4850407a55…`** on the full candidate plus your patch (the reviewer reproduced it).

## 3. Verification on the integration candidates

Two throwaway candidate branches were built in their own worktrees. Neither is merged anywhere.

| | `candidate/orch-742-integration` (A+B+C) `b9b19c06` | `candidate/orch-742-integration-bc` (B+C) `59e31724` |
|---|---|---|
| SMOKE (mining off) | **508 passed** | **508 passed** |
| Chunk D (split-process gate) | **187/187, failed=none** | **187/187, failed=none** |
| mining suite | 226 passed | — |
| reaction evidence (model / view / hook) | 19 / 9 / 1 | 19 / 9 / 1 |
| eval_v2 | 60 passed, 1 xfailed | 60 passed, 1 xfailed |
| gold-readers (22 files) | 465 passed, 8 skipped | 465 passed, 8 skipped |
| reviewers' extra selection (24 files touching `streamlit_app.py` or scanning `src/`) | 2 failed / 361 passed / 3 skipped **+ 4 solo files all green** | identical |
| isolated collection (`T2PW_ISOLATED_COLLECT_ALL=1`) | 226/226 files | — |
| G11 check | 52 reports, 0 non-compliant | 44 reports, 0 non-compliant |

**The 2 extra-selection failures predate this work and are identical at base `9f13188e`.** No lane introduced
them, and neither is recorded anywhere in the sprint docs:
`test_rag_seed_entity_reimport.py::test_the_orchestrator_passes_the_merge_base_to_conform` (a `StopIteration` at
`:590`) and `test_c075_source_support_armed.py::test_corpus_the_armed_pass_over_every_committed_artifact_and_its_own_paper`
(an unexpected catch, `('PMC12096016','strict',"Pantetheine 4'-phosphate")`).

## 4. Why mining should NOT be integrated now

Both independent candidate reviewers returned `NOT_READY` on the full candidate, **for governance reasons, not
for a runtime defect**:

1. **Unreviewed scope.** `ORCH-742a` approved only `9c085967..23e1ae83` (the two fixes). But the branch's history since the sprint base (`91550546`, `f6f9db53`, `3bbb6b76`, `9c085967`) is the whole experimental subsystem: about 17.6 k lines, plus the `streamlit_app.py` hook. No sprint reviewer has approved that code, and `DECISIONS.md` / `MASTER_PLAN.md` / `PRODUCT_CONTRACT.md` never mention mining.
2. **The pilot says the feature is not yet evaluable** (§6).
3. Housekeeping. The lane's pin verdicts sit next to its reports, not under `g11/pin/<task>/`, so the rule-10 checker counts them as unpinned. The pilot record's `"api_key_present": true` also trips `tests/test_mining_artifacts.py:86-94` (a mismatch between test and implementation).

## 5. Recommended integration sequence

This differs from the requested order (mining → UI → eval), for the reasons in §4:

1. **`eval/reaction-rag-metrics` @ `a51e9609`.** New files only; no production file and no dirty file in the primary checkout is touched, so it merges in place. Before merging, decide the **PMC12657337** flag. The final bio-audit certifies 3 of the 4 new `supported_reactions_complete=true` cases. For PMC12657337 it says the lumped FPP + IPP + DHNA → MK-7 conversion (Results 3.2) is still unbooked. Under D-087, an uncertified flag should be set to `false` or certified. Also register `tests/test_eval_v2_report.py` in `TEST_MATRIX` as a repo-wide census pin. It fails, by design, the moment a new run leg or a new cache blob is committed, **including the upcoming results run**.
2. **`feature/reaction-evidence-ui` @ `e328961e`,** after a **product-owner D-xxx** authorizing its 8-line EOF hook in `streamlit_app.py` (frozen and PROTECTED per D-090 / D-097 §4 / D-102). The hook emits only `st.info` or markdown, and `batch/driver.py` classifies legs from error, warning and exception elements only, so leg classification cannot change. The reviewer checked this. **Mechanics:** `git merge` in the primary checkout will refuse while your protected edit is uncommitted in that file. The edit must be set aside **from the byte-exact patch copy** (never discarded), the merge made, and the patch re-applied. The result should hash to the new pin you record. This step changes a protected file, so it waits for you.
3. **`experiment/pathway-mining`: not now.** First: a product-owner decision to bring the subsystem onto the sprint line, an independent review of `9f13188e..23e1ae83` in full, a fix for at least pilot findings F1–F3, and relocated pins. The pilot record belongs on the experiment branch. Fast-forward `experiment/pathway-mining` to `dcb64fc4` if you want it there.

After any authorized merge: run SMOKE, Chunk D, each lane's focused suite and gold-readers again, on the real branch.

## 6. Pilot outcome (full record: `experiment/pathway-mining-pilot-001` → `docs/pathway_mining/pilot/001/PILOT-001-RESULT.md`)

One pre-registered live run (config frozen at `08db8199` before any model call): jasmonate biosynthesis,
*A. thaliana*, seeds 12-OPDA + AOS, branch-default budgets. It ran 304 s at an **estimated $0.0034**, with 0
survivors, and **was not re-run**.

- **21 papers screened, 1 read:** a *Marchantia* review, typed `unknown`. **0 reactions** entered the target graph. **19 candidates** were all held back, correctly.
- **`pwml-bio-auditor`: NOT_PLAUSIBLE** as an Arabidopsis expansion. No reaction was fabricated, but the run produced Arabidopsis evidence for 0 canonical steps. Of the 19 candidates, 11 are fatty-acid desaturation or elongation, and 8 are the wrong species.
- **Key defects**, registered and not fixed:
  - **F1:** every enzyme, gene and species field was dropped as uncited (59/59), so no claim could ever reach the target species.
  - **F2:** round-2 retrieval collapsed: 600 result slots held 20 unique papers (query formulation, top-10 cap with no paging, one-attempt exhaustion).
  - **F3:** species labels are reached by fall-through.
  - **F9:** the mining CLI never loads `.env`. The pilot used a launcher that calls the project's own `ensure_dotenv_loaded()`; no mining code was changed.
- **Live confirmation:** preflight passed with the new 64-token probe.

**Verdict: pathway mining is BLOCKED for evaluation.** It is safe (it invents nothing, and the species gate
held), but it cannot yet produce the target-species evidence it exists to find.

## 7. Evaluation-readiness outcome (`docs/evaluation_v2/READINESS.md` on the eval branch)

**READY WITH GAPS.**

**Delivered:**
- 4 newly complete gold cases in a new versioned file; `pinned_v1.json` is byte-identical.
- Reaction P/R/F1, the unsupported rate, and substrate/product/enzyme/species accuracy, with minor cofactors kept separate.
- Evidence support and per-identifier mapped-ID correctness. It now **detects** "claimed SAM, mapped to another molecule".
- Live retrieval Recall@1/3/5, P@5, MRR, and the A/B/C/D failure split.
- The un-gated RAG influence audit, and a reproducibility-completeness table.

**Sample numbers** (new artifacts, never a rescore of T-107/108/109): canonical legs P 86.0 % / R 96.6 % / F1 91.0 %,
unsupported 2.8 %; live Recall@5 44.4 %, MRR 0.293.

**Gaps:**
- P/R/F1 rests on only 4 papers (3 certified).
- 6 of 11 reproducibility fields are not recorded by production. Closing that needs production hooks: a product decision.
- `wrong_identity` is a candidate list: hand precision is 5/21.
- Negative-query rejection is 0/49 by construction, because there is no score floor.
- The **fallback-embedding cache defect is confirmed**: lexical vectors are cached under the real model key. It is pinned as a strict xfail and not fixed.
- The mapped-ID reference is pinned to the caches' HEAD blobs.

## 8. Do NOT merge

- `experiment/pathway-mining`, until §5.3 is done. Merging brings the unreviewed subsystem.
- `experiment/pathway-mining-pilot-001` into the sprint: it is the pilot record, 2.6 MB.
- `candidate/orch-742-integration` and `candidate/orch-742-integration-bc`: verification scaffolding only. Merge the lane branches themselves in the §5 order.
- The protected `streamlit_app.py` edit (never commit it), and the dirty `data/*cache*.json` files (committing them breaks the no-cache rule **and** `test_eval_v2_report`'s HEAD-blob check).

## 9. New findings — registered, NOT chartered

| id | finding | where |
|---|---|---|
| pilot F1–F10 | see the pilot result | `docs/pathway_mining/pilot/001/PILOT-001-RESULT.md` |
| X-1 | two test failures that predate this work and are unrecorded (§3) | `test_rag_seed_entity_reimport.py:590`, `test_c075_source_support_armed.py:879` |
| X-2 | fallback-embedding cache stores lexical vectors under the real model key (confirmed) | `tests/test_eval_v2_embedding_cache.py:53` (strict xfail) |
| X-3 | `pipeline.py:1322` prints "LOCKED REACTION FILTER" for `quarantined_locked_reactions` rows that were all written by `process_normalizer.dedupe_processes`. Frozen production; not touched | `pipeline.py:1322` |
| X-4 | mining hook's `try/except ImportError` also wraps the render and `st.stop()`, so an import error inside the mining page is swallowed and both pages render | `streamlit_app.py:151-157` (mining lane) |
| X-5 | the evidence section is at EOF, after the pipeline form's `st.stop()` calls, so on a rerun that stops early it waits for the next rerun | `streamlit_app.py` EOF (UI lane) |
| X-6 | `max_new_reactions_per_round` is enforced as a **stop** between rounds, not a throttle within one. This matches the brief and the docs; whether it should be a throttle is your call | `mining/budget.py:217` |
| X-7 | `.claude/settings.json` changed during this session (harness permission-allowlist entries). Its diff against HEAD also *removes* `git push/reset/rebase/merge/clean` rules; whether that removal predates this session could not be determined | primary checkout, user-owned |
| X-8 | `docs/pathway_mining/cli.md` puts `--enable-pathway-mining` after `run`; it is a top-level flag | mining docs |

## 10. Left on disk (nothing pruned)

- **Worktrees:**
  - `C:/t/mining`, `C:/t/evidence-ui`, `C:/t/eval`: the lanes.
  - `C:/t/mining-base`, `C:/t/sprint-base`: read-only bases.
  - `C:/t/mining-pilot`: on `experiment/pathway-mining-pilot-001`.
  - `C:/t/integration`, `C:/t/integration-bc`: the candidates.
- **Scratch:**
  - `C:/t/mining-pilot-runs`: the pilot run, the launcher and the audits.
  - `C:/t/eval-fulltext`: the fetched papers, sha-pinned in the gold file.
  - `C:/t/review-evidence`: reviewer re-runs.
  - `C:/t/protected/user_streamlit.patch`, `C:/t/bt`.
- **New branches:** `feature/reaction-evidence-ui`, `eval/reaction-rag-metrics`, `experiment/pathway-mining-pilot-001`, `candidate/orch-742-integration`, `candidate/orch-742-integration-bc`. `experiment/pathway-mining` gained 2 commits.
