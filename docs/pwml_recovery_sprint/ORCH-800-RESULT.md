# ORCH-800 - unmerged-branch audit (read-only)

Base: `main` == `origin/main` == `sprint/pwml-recovery` == `b25327c1`. Method: `git branch --no-merged main`, `git cherry main <br>`, `git log main..<br>`, grep of `LEDGER.md`, `DECISIONS.md`, `ORCH-*-RESULT.md`. No git write operation, no Python, no tests. 34 unmerged branches.

Counts: **A 3, B 14, C 17, D 0.** No branch qualifies as "approved but never integrated".

## 1. Table

| branch | last commit | ahead | class | evidence |
|---|---|---|---|---|
| rag-contract-fixes | 2026-07-27 | 1 | A | `git cherry`: the one commit is `-` (patch-equivalent, already in main) |
| orch780/support-per-row | 2026-10-26 | 5 | A | LEDGER: "ORCH-780 (D-116 s1) cherry-picked from `orch780/support-per-row` `6b5e33b1`"; main has `8a32f02d` and merge `e37617c6`; 4 of 5 commits are `-`, the `+` is `6b5e33b1` itself (the cherry-picked content under another SHA); G9 `evidence/g11/ORCH-780/06,07` |
| candidate/orch-742-integration-bc | 2026-09-23 | 3 | A | 742b `e328961e` and 742c `a51e9609` are ancestors of main; the one `+` commit (`59e31724`) is only G11 evidence under `evidence/g11/ORCH-742d/` |
| agent/c114-d088-diagnostics | 2026-09-02 | 1 | B | LEDGER: "C-114 NOT MERGED, superseded in shape"; `RULING-C114-DISPOSITION.md` |
| agent/p42b-legdeadline-wiring | 2026-08-14 | 1 | B | tip `ec3db99f` "C-042b: BLOCKED on the product question"; FINDINGS P2-04, ORCH-758-RESULT lines 225 and 306 (no production LegDeadline caller; handled by D3) |
| agent/o1-reconciled-audit | 2026-08-06 | 3 | B | 3 docs-only "O-1 audit" commits; no mention in LEDGER or any RESULT doc (see Not settled 1) |
| backup-pwml-complex-broken | 2026-05-20 | 2 | B | pre-sprint; commit subjects "please bro work plez"; name says broken; no doc reference |
| improving-oneshot-ability | 2026-06-12 | 7 | B | pre-sprint; tip "bunch of stuff butits ocmpltely brken"; no doc reference |
| improving-oneshot-ability-clean | 2026-06-12 | 6 | B | same line, pre-sprint; no doc reference |
| card/C-094-f134 | 2026-08-27 | 2 | B | LEDGER wave record C-094..C-098: complete but blocked on companion C-098; affected rows never reach export |
| card/C-098-f135 | 2026-08-27 | 3 | B | same wave record: "uncommitted partial"; refusal ruled |
| card/C-098a-cap | 2026-08-27 | 3 | B | LEDGER: "Held. Inert and harmless, but merging it alone adds capability with no producer" |
| card/C-098b-gate | 2026-08-27 | 4 | B | LEDGER "What C-098b found"; held with C-098a |
| orch776/audit-keeps-paper-enzyme | 2026-10-26 | 6 | B | `ORCH-776-RESULT.md` exists only on the branch; LEDGER `cdb43ddb` "lanes 776-780 paused"; no reviewer approval in main docs (Not settled 2) |
| orch777/enzyme-retrieval-placeholder | 2026-10-26 | 3 | B | tip says "UNMEASURED"; same pause; no approval (Not settled 2) |
| orch778/keep-passing-enzymes | 2026-10-26 | 6 | B | only the implementer record (SMOKE 511, selection 498); same pause; no approval (Not settled 2) |
| orch779/regulators-not-enzymes | 2026-10-26 | 5 | B | `ORCH-779-RESULT.md` on the branch: "Not done: no measurement and no test run", open Q1-Q3; no approval (Not settled 2) |
| candidate/orch-742-integration | 2026-09-23 | 11 | C | LEDGER "ORCH-742 close ... integration PROPOSED, not performed". 742b/742c already in main; the remainder is 742a, which lives on `experiment/pathway-mining` |
| experiment/pathway-mining | 2026-10-21 | 217 | C | the mining line; ORCH-745/746 RESULT and LEDGER: nothing merged into sprint or main; product-owner decision pending |
| experiment/pathway-mining-pilot-001 | 2026-09-23 | 7 | C | pilot snapshot of the mining line (ORCH-742/743 RESULT) |
| experiment/pathway-mining-pilot-002 | 2026-09-24 | 25 | C | pilot snapshot (ORCH-743 RESULT) |
| experiment/pathway-mining-pilot-003 | 2026-09-24 | 33 | C | pilot snapshot (ORCH-743/745 RESULT) |
| experiment/pathway-mining-pilot-004 | 2026-10-20 | 138 | C | pilot-004 recovery line (ORCH-746/748/771 RESULT) |
| orch745/a-evidence-hygiene | 2026-10-07 | 49 | C | ORCH-745-RESULT line 197: "(merged)" into `experiment/pathway-mining`, not into main; review r2 APPROVE_WITH_FINDINGS |
| orch745/b-species-reading | 2026-10-15 | 117 | C | species gate "not merged"; ORCH-745 and ORCH-746 closed BLOCKED (LEDGER "ORCH-746 close") |
| orch745/c-run-control | 2026-10-07 | 43 | C | ORCH-745-RESULT line 198: "(merged)" into the mining line; approved after r1; not in main |
| orch760/a-extraction-capacity | 2026-10-20 | 150 | C | LEDGER: brief `docs/pathway_mining/evidence/orch760/BRIEF.md` on the mining branch; lane of the experiment line |
| orch760/b-outreach | 2026-10-20 | 151 | C | same |
| orch760/c-discovery-graph | 2026-10-20 | 156 | C | same |
| orch760/d-link-novelty | 2026-10-20 | 193 | C | same |
| orch760/d2-novelty-followup | 2026-10-20 | 200 | C | same |
| orch760/e-link-view | 2026-10-20 | 203 | C | same |
| orch760/f-throughput | 2026-10-21 | 202 | C | same |
| orch760/h-organisms | 2026-10-21 | 200 | C | same |
| orch765/parenthetical-bridge | 2026-10-24 | 15 | C | LEDGER 2026-10-23 "BLOCKED on a product-owner ruling" `835e374d`; D-110 amendments 1-2; round 2 handed to the sub-orchestrator; no approval of a final tip found |

## 2. Worktree counts (from `git worktree list`)

- Under `.claude/worktrees/`: 193 lines matched. Under `C:/t`: 57 lines matched. Prunable per `--porcelain`: **0**.
- Worktrees on the 34 unmerged branches: 27 (18 under `C:/t`, 9 under `.claude/worktrees`).
  - On class A or B branches: **9** (`.claude/worktrees`: o1, p42b, c114, orch776, 777, 778, 779, 780 = 8; `C:/t/integration-bc` = 1).
  - On class C branches: 18 (17 under `C:/t`, plus `.claude/worktrees/orch765`).
- 120 worktrees sit on branches already merged into main (counted by branch name; may include the main checkout). They are removal candidates, not merge candidates.
- Prune candidates: **0 git-prunable**; 9 on A/B unmerged branches plus up to 120 on merged branches are removal candidates.

## 3. Recommended actions

1. Nothing to merge: no class D branch.
2. Class A (3): safe to delete after their worktrees (`orch780`, `C:/t/integration-bc`) are removed; no content is lost.
3. Class B pre-sprint (`backup-pwml-complex-broken`, `improving-oneshot-ability*`): delete if the owner agrees; they predate the sprint.
4. Class B cards and agents (`card/C-094*`, `C-098*`, `agent/c114*`, `agent/p42b*`): LEDGER keeps them "as the record"; the record is in the docs. Tag first, then delete, is a cheap safeguard.
5. Class C: never merge or delete as a side effect. One product-owner call is needed on the mining line (merge it, or archive it under a tag). Merging brings 217 commits of base history.
6. `orch776`-`orch779`: decide whether the lanes resume. D-115 is locked and ORCH-780 has landed; 776-779 implement D-115 sections but have no reviewer approval.

## 4. Not settled

1. `agent/o1-reconciled-audit`: three docs commits about an "O-1 audit", unreferenced in LEDGER and RESULT docs. Put in B because it could not be decided; its audit table may be worth keeping. Look before deletion.
2. `orch776`-`orch779`: not rejected, but paused (`cdb43ddb`) and unreviewed in main docs. I did not search branch trees for REV-77x documents beyond the RESULT files. Class D requires approval evidence and none was found.
3. `candidate/orch-742-integration`: classed C because 742a is on the mining line. If the owner only wants 742b/742c, they are already in main and the branch is class A.
4. Worktree counts come from line matches in `git worktree list`; the `.claude/worktrees` figure of 193 may include a row for the primary checkout. Dirty state of worktrees was not checked.
5. orch760 evidence was read only through LEDGER mentions; per-lane review approval was not examined.
