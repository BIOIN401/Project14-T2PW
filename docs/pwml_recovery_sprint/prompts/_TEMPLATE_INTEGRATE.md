# Integration prompt template — `M-xxx`

Executed by the Lead Orchestrator, which writes no implementation code. One merge per
invocation.

---

```
[S1] [S5]

ROLE
  Integration Authority executing ONE merge.

BRANCH             agent/<name>
DEPENDS            <merged branch IDs>
REVIEWER VERDICT   APPROVE (from V-xxx by <agent id>)

DO — in this order; stop at the first failure. NO REBASE at any step, and never
merge integration back into a worker branch.
  1. Confirm EVERY dependency is already in sprint/pwml-recovery.
     If not -> BLOCKED. Do not merge. Update the ledger and stop.
  2. Independently review the EXACT card tip. Then on sprint/pwml-recovery:
     git merge --no-ff --no-commit <reviewed-tip>
  3. Inspect the prospective staged merge against the card's authorized path
     manifest. Anything outside it -> stop; do not commit the merge.
  4. Run the card's focused tests and required gates ON THAT COMBINED
     PROSPECTIVE INTEGRATION STATE. This is where an earlier merge that
     invalidated this card's assumptions is caught -- which is why it happens
     now and not at dispatch. If they fail -> FREEZE the prospective merge state
     and report the affected lane. Do NOT reset, abort destructively, rewrite
     the worker branch, or commit a failing merge.
  5. If they pass, commit the merge with the required message.
  6. Run the SMOKE SUITE (460 tests, ~40 s) on the integration branch.
     If it fails -> revert the merge immediately, send to CORRECTION.
  7. If the branch is chunk-D-marked in TEST_MATRIX.md, run the COMPLETE chunk D
     gate (177 tests, 9-13 min). TEST_MATRIX.md § Chunk D cadence says which
     cards may defer the 23-node qb cohort to a pack-level checkpoint.
  8. Record the new integration SHA in LEDGER.md and set the row to ACCEPTED.
  9. Move every branch whose last dependency this was from BLOCKED to READY.

MERGE GATES -- all must hold. Any failure stops the merge.
  G1  dependencies merged
  G2  diff within boundary
  G3  focused tests pass
  G4  affected tests pass, or a pinned baseline moved deliberately with an
      exact documented delta
  G5  independent reviewer approved the actual diff
  G6  no biological gate weakened to increase PWML production
  G7  incomplete-but-correct pathways still exported as review_required
  G8  no exporter repairs biology after the freeze
  G9  a claimed correction or preservation of PRE-EXISTING OBSERVABLE BEHAVIOUR
      has a proof that FAILS BEHAVIOURALLY at the base and passes at the tip;
      symbol absence is not proof. A genuinely NEW capability or module instead
      carries an EXPLICITLY LABELLED new acceptance test and needs no fabricated
      base failure. Mislabelling a regression as new functionality is a REJECT.
  G10 smoke passes after the merge, on the integration branch
  G11 test-process lifecycle: every command in the branch's evidence ran through
      the bounded foreground wrapper, and the cleanup report shows FINAL
      SURVIVING COUNT = 0. Survivors make the run an INFRASTRUCTURE FAILURE, not
      a test result -- it cannot satisfy G3, G4 or G10. If any agent reports a
      survivor, STOP ALL DISPATCH and record PID, command line, start time and
      memory in the ledger before continuing.

REPORT
  ## MERGED                    yes/no
  ## NEW INTEGRATION SHA
  ## MERGE CONFLICTS           what, and how resolved
  ## PARENTAGE PROOF           2nd parent == reviewed tip SHA exactly; and the
                               owned-path first-parent..merge diff == the
                               reviewed card diff from its dispatch base
  ## FOCUSED TESTS ON THE PROSPECTIVE MERGE STATE   pasted
  ## SMOKE POST-MERGE          pasted (expect 460 passed)
  ## CHUNK D                   pasted, if applicable
  ## GATES                     G1..G11, one line each
  ## BRANCHES NOW UNBLOCKED
  ## ASSUMPTIONS INVALIDATED   for any still-pending branch
  ## LEDGER UPDATED            yes/no
```

---

## Rejection triggers

Do not merge; send to CORRECTION or REJECT:

- an earlier merge invalidated this branch's assumptions — proven by rerunning the card's
  focused tests on the **prospective combined integration state**, never by rebasing
- the diff is correct but out of boundary
- the report claims a runtime behaviour with no pasted evidence
- a benchmark failure justifies code without an adjudication of
  `product_contract_violation` | `gold_data_defect` | `policy_disagreement`, citing the
  gold `relevance_note` / `export_rationale`
- the agent resolved an open question from `DECISIONS.md` § Open on its own authority

## Parallel branches — what the merge must prove

For the **second or later parallel branch cut from an earlier common base**, whole-tree
equality between the merge and the standalone reviewed tip is **neither required nor
possible**. Require instead, all three:

1. the merge's **second parent equals the independently reviewed tip SHA exactly**;
2. within the card's **owned paths**, the **first-parent-to-merge diff equals the reviewed
   card diff from its dispatch base**;
3. any remaining tree content comes from the merge's **first parent** — the already
   authorized integration history — and **not** from new unreviewed paths.

This **supersedes forward** any interpretation requiring the whole merge tree to equal a
standalone parallel tip. D-022's historical record and evidence are unchanged.

## On return from an overnight run

Do **not** merge everything that finished. Review in **dependency order**. For each branch
take the `--no-commit` merge and rerun its focused tests on that combined prospective state
before applying G1–G11 and committing. **Never rebase, and never merge integration back
into a worker branch.** A branch whose assumptions an earlier merge invalidated is
rejected, not patched in place.
