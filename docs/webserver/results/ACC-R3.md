# ACC-R3 + FIX-NONAME: verifier r3 and the post_audit nameless-entity fix, merged and live

2026-11-24. PO approved both ("do whatever, just don't break anything").

## Outcome

| branch | commit | what |
|---|---|---|
| main | `5bf9d552` | `1a52fb55` + `acc/verifier-r3` `460c3804` + `fix/post-audit-empty-name` `767c6e66` |
| webserver/refactor | `2f7faea4` | main merged in |
| deploy/local | `f717e0ae` | webserver/refactor merged in; the server was restarted on it |

The verifier stays ON for web jobs. Typing and VD-18 stay off.

## What changed

**Verifier r3** (flag-gated, `service/reaction_verifier.py`; lane record `docs/pwml_output/ACCURACY.md` §15–§15a):
- **Fold instead of refuse.** A rename that F-G3 refuses now folds onto an existing row only when the participants
  match by `reference_key`. This removes the duplicate EntE/EntF pairs that 3-draw jobs produce.
- **Spelling key no coarser than `reference_key`.** This fixes an R6 B2 bug that was live on main: an added β-alanine
  was written as "alanine".
- **Fold target needs an equal `reference_key`** (R6 B1: α/β-carotene could merge).
- **Orphan-entity scan covers every reference,** including enzymes, transporters and complex components (R6 N1).

**FIX-NONAME** (`curation/apply_audit_patch.py`, `service/postpipeline.py`; record `results/FIX-NONAME.md`):
- **Before:** about 1 in 17 Gemma runs aborted at post_audit with `entity_missing_name`.
- **Fix:** an audit or enrichment op that would leave an entity nameless is now refused; the row is kept and the reason
  is recorded.
- **N1 tightening:** later ops under a refused op's path are refused with it.
- **Diagnostics:** `mapped_failure_snapshot.json` gains `failing_issues`.

## Evidence (`docs/webserver/evidence/ACC-R3/`)

**Reviews**
- Verifier: R6 REJECT, then R7 APPROVE (`REVIEW-R6.md`, `REVIEW-R7.md`).
- FIX-NONAME: APPROVE WITH NOTES, and the N1 re-check APPROVE WITH NOTES (`FIX-NONAME-REVIEW.md`).

**Replays**
- Verifier: 0 gold lost on every stored answer set, and 14 new folds, all EntE/EntF.
  - Enterobactin precision 0.844→0.878 and 0.867→0.902 (deepseek); 0.75→0.833 on the server job.
- FIX-NONAME:
  - real audit patches 80/80 identical, gap-resolver patches 6/6, deterministic patches 119/119;
  - Gemma-shaped ops 119/119 aborted before and now continue;
  - 0 refusals on any real patch.

**Gates**
- Main merge `5bf9d552` (`main-merge-gates.log`):
  - SMOKE 511, golden 2/8/4/24, equivalence 8/14/6, vision 370 (+1 skip), web 83, LLM client 35;
  - focused 1207 passed, 1 failed: the stale conform test, which exists only on main and is fixed on webserver/refactor;
  - 0 survivors.
- Deploy tree `f717e0ae` (`deploy-tree-gates.log`): the same suites green, except the 2 known
  `test_streamlit_quarantine_boundary` failures. Those come from this tree's `.env` reaching the live DB (recorded in
  `ACC-VERIFIER-R2.md`).

**Live server check** (`live-check-item5.txt`, R7 list)
- Job `a4abec03…`, enterobactin, PMID 40400396, Gemma, 3 draws, work `item-5`.
- Result:
  - a PWML was produced (needs review);
  - EntE and EntF each appear once, with no duplicate reaction signature;
  - no `PWML_IR_DUPLICATE_NAMED_ROW` and no `AMBIGUOUS_RENAME_TARGET`;
  - all 5 gold steps are present (as in attempt 1), and there were no renames or additions, so 0 `reference_key`
    mismatches;
  - 1 lock entry was removed, matching the 1 merge (a partial EntB row folded into the full one);
  - no dropped entity is still referenced.
- Verifier: 1 call, 4.6 s, no error.
- This draw used the canonical "2,3-dihydroxybenzoyl-AMP" spelling, so the new fold path did not trigger live. It is
  covered by unit tests and the replay.

## Dead ends

- R6 rejected the first r3 (`d5d0ec95`), which could fold β-carotene into α-carotene. Fixed in `f26c7f17`.
- The FIX-NONAME covering run put 44 files in one process and stalled silently while holding the heavy lock. It was
  killed by PID and re-run in timed groups.
- FIX-NONAME's root cause is inferred: no raw Gemma responses were saved.

## Open questions and follow-ups

- **PO:** keep the raw audit and enrichment LLM responses in run folders, so the exact op behind the next crash can be
  found?
- **PO:** try the tRNA Glu one-liner (Bug B, `audit_json_llm.py:680-684`) behind a flag, measured first on
  nucleic-acid papers? It hits 1 of 62 stored runs.
- **PO, biology:** the lactate false positive. Stage 1 marks the LDH assay out of scope, then Stage 2's `pwml_infer`
  re-adds it with no lineage (`ACCURACY.md` §15). The fix would be in the Stage 2 merge.
- **FIX-NONAME notes:**
  - N2: the re-run path is proven only on synthetic shapes.
  - N3: the curator and focused-repair callers are not replayed.
  - N4: no test pins the snapshot keys.
  - After a refused insert at index i, a later edit to the old row i in the same batch is also refused.
- **Verifier:** NAD+ and NAD no longer share a spelling key, so an abbreviated co-product addition can be refused.
  The measured cost is zero; an alias would fix it if it shows up.
- Worktrees for the PO to remove: `C:/t/acc-old`, `C:/t/acc-ver5`, `C:/t/acc-r3base`, `C:/t/acc-r6base`,
  `C:/t/merge-acc2`, `C:/t/ws-acc-check`, `C:/t/merge-r3`, `C:/t/fix-noname`.
