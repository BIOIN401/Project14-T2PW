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
