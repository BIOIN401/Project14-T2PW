# PROGRESS (running log, PWML output assessments)

Branch `pwml/output-assessment` in worktree `C:/t/oa`, cut from `85b5f9d4`. Scratch: `C:/t/ev-oa/`.
Budget: at most 8 dispatches, at most 3 at once. Docs only, no product change.

## Resume instructions
All work finished. Nothing to resume. If anything is amended: edit docs here only; PLAN.md section 1 numbers come from assessments 01-07b and VERIFICATION.md.

## Dispatch log
| # | name | model / effort | scope | status |
|---|---|---|---|---|
| 1 | funnel | sonnet / high | assessments 01 and 03 | done: 01-content-loss-funnel.md, 03-lost-content-by-kind.md, evidence/01 |
| 2 | mapping | sonnet / high | assessment 02, online lookups | done: 02-mapping-failures-and-homology.md, evidence/02 |
| 3 | payload | sonnet / high | assessment 06 (formats) | done: 06-payload-writer-and-format-requirements.md, evidence/06 |

Wave 1 finished (3 dispatches). Dead ends: funnel agent ran two stray empty `python -` heredocs, stopped within seconds, no survivors. Next: wave 2 = assessments 04+05, 07, KEGG licence; wave 3 = verifier; then PLAN, CLEAN-MODEL, DECISIONS-PROPOSED, commit.
| 4 | accvar | sonnet / medium | assessments 04 and 05 | done |
| 5 | import | sonnet / medium | assessment 07 | done: 36 of 61 pass today QA |
| 6 | kegg | sonnet / medium | assessment 07b | done |
| 7 | verify | sonnet / medium | recompute 3 numbers | done: claims 1 and 3 agree, protein claim disagreed (definition) |
| 8 | reconcile | sonnet / medium | protein count reconciliation | done: 168/435 adopted, 02 appended |

Total 8 dispatches: 3 sonnet/high, 5 sonnet/medium. Max concurrency 3. Dead ends: stray empty python heredocs in dispatches 1 and 4 (stopped within seconds). PathBank MySQL not read (gap). Orchestrator wrote PLAN.md, CLEAN-MODEL-REQUIREMENTS.md, DECISIONS-PROPOSED.md and committed.


---
# Clean-model build (branch `pwml/clean-model`, worktree `C:/t/cm`, from `075d127d`)
Scratch: `C:/t/ev-cm/`. Budget: at most 10 dispatches, one implementer edits the worktree at a time. Additive only (P-14).

## Resume instructions
Read this log, then `CLEAN-MODEL.md` (once it exists) and `git log`. Next undone row in the dispatch log is the next step.

## Dispatch log (clean model)
| # | name | model / effort | scope | status |
|---|---|---|---|---|
| 1 | A model+builder | sonnet / high | model.py, builder, vocab, roundtrip, build_all | done: 152/152 built, 152/152 exact round trip, 36 tests pass; title 0/152 locally (online layer must fill); PMID 152 |
| 2 | B online fill | sonnet / medium | fill/ (http, cache, sources, fill), fill_sample.py | done: 65 tests pass (A+B); ChEBI name search via OLS4 (ChEBI API has no exact search); not yet run live at scale |
| 3 | C SBML exporter | sonnet / medium | export/sbml.py, sbo.py, validate_sbml.py | done: 74 clean-model tests pass; libsbml 6 runs: 0 errors, only 80601 (no initial value, deliberate) |
| 4 | M measurement | sonnet / medium | MEASUREMENT.md; build 152, SBML 152 (0 errors), live fill 10 runs | done: found 2 wrong + 2 questionable reaction EC/Rhea fills (catalyst-EC inference) |
| 5 | K KGML + fix | sonnet / medium | export/kgml.py, kgml_all; reaction EC/Rhea fill made opt-in (default off) | done: 84 clean-model tests pass; 152/152 KGML files pass structural check |
| 6 | review | opus / high (ws-reviewer) | independent review of diff 075d127d..37416257 | APPROVE WITH FIXES: 3 major + 1 major (KGML reaction ids, from recall) + 11 minor |
| 7 | F fixes | sonnet / medium | 13 items; Biopython check of KGML (0 -> 152 of 152 parse) | done: 102 clean-model tests pass |
| 8 | re-review | sonnet / high (ws-reviewer) | delta 37416257..83f5bd74 | APPROVE WITH FIXES: 13 of 13 fixed (spelling of ubiquitination unverified); placeholder in CLEAN-MODEL section 7 fixed by orchestrator |
| 9 | test runner | haiku / medium | focused, SMOKE, golden at 83f5bd74 | done: focused 101 passed + 1 skipped (102 with T2PW_RUNLIST), SMOKE 511 passed, golden 38 passed, 0 survivors |

Total 9 dispatches of 10: sonnet/high x2 (A, re-review), sonnet/medium x5 (B, C, M, K, F), opus/high x1 (review), haiku/medium x1 (runner). No scouts. All work finished; nothing to resume. Record: CLEAN-MODEL.md.


---
# Final payload, phase 1 (branch `pwml/final-payload`, worktree `C:/t/fp`, from `9ae702ba`)
Scratch: `C:/t/ev-fp/`. Budget was at most 12 dispatches; about 15 were used (five lost to environment stalls). One implementer edited the worktree at a time. Everything behind `T2PW_FINAL_PAYLOAD` (default off). Record: `FINAL-PAYLOAD-P1.md`. Code HEAD `f83bc613`.

## Dispatch log (final payload P1)
| # | name | model / effort | scope | status |
|---|---|---|---|---|
| 1 | FP-1 draft | sonnet / high (general-purpose) | `service/final_payload.py`, seam in `postpipeline.py`, tests | draft written; stopped on a write denial (setting `blockWritesOutsideWorkingDirectories`), no tests run |
| 2 | FP-1 resume (message) | sonnet / high | same agent | refused to write through Bash on another agent's word; nothing changed |
| 3 | FP-1 finish | sonnet / medium | `_plan` guard, tests | guard applied; tests blocked 1 h by a stale heavy lock; machine then shut down overnight |
| 4 | FP-1 tests (message) | sonnet / medium | fp1, fp1cm, fp1g | 17 + 101 (+1 skipped) + 38 passed; committed `15a5c0de` |
| 5 | replay harness + base0 | sonnet / medium | `C:/t/ev-fp/replay/`, base `9ae702ba`, flag off | 152 runs, 97 write a PWML |
| 6 | FP-2 title and references | sonnet / high | `final_payload.py`, `pwml/ir.py`, `pwml/writer.py`, `service/pathway.py`, `postpipeline.py` | 131 + 141 passed, flag off 97/97 byte-identical; `a01f7076` |
| 7 | flag-on replay (message) | sonnet / medium | on1 | agent looked hung (waiting at the RAM gate); stopped waiting, redispatched, it later finished |
| 8 | flag-on replay + measurement | sonnet / medium | on1 | 709 live requests, 0 failures, 596 ids |
| 9 | hand check | sonnet / medium | 31 ids, 6 titles, renames | 2 wrong of 31 (MK, PMEA; CDG outside sample) |
| 10 | FP-3 (2 messages) | sonnet / high | short-abbreviation rule | 136 passed; `aafb7f17` |
| 11 | review | opus / high (ws-reviewer) | `9ae702ba..aafb7f17` | APPROVE WITH FIXES (F1, F2 major) |
| 12 | FP-4 fixes | sonnet / high | F1 to F4, F6 to F8 | 143 + 141 passed; `f83bc613` |
| 13 | final tests | haiku / low (ws-test-runner) | SMOKE, golden, focused at `f83bc613` | 511, 38, 143 passed, 0 survivors |
| 14 | re-review | sonnet / high (ws-reviewer) | `aafb7f17..f83bc613` | APPROVE |
| 15 | PathWhiz import pairs | sonnet / medium | harness, 24 pairs off vs on | identical error sets, 24 references created |

Dead ends: write denial under `C:/t` until the restart; a correct refusal to write on another agent's word; heavy lock held by the import-harness builder; RAM below the floor (Docker), floor lowered to 4 GB for focused runs; an empty `python -` heredoc once (exited at once); a stale agent deleted replay work dirs (one run re-run).
