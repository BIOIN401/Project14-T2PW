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

DO — in this order; stop at the first failure
  1. Confirm EVERY dependency is already in sprint/pwml-recovery.
     If not -> BLOCKED. Do not merge. Update the ledger and stop.
  2. Rebase agent/<name> onto the CURRENT integration commit.
  3. Re-run the branch's focused tests AFTER the rebase. An earlier merge may
     have invalidated this branch's assumptions -- that is precisely why the
     rebase happens now and not at dispatch. If they fail -> CORRECTION.
  4. Merge into sprint/pwml-recovery.
  5. Run the SMOKE SUITE (457 tests, ~40 s) on the integration branch.
     If it fails -> revert the merge immediately, send to CORRECTION.
  6. If the branch is chunk-D-marked in TEST_MATRIX.md, run chunk D (222 s).
  7. Record the new integration SHA in LEDGER.md and set the row to ACCEPTED.
  8. Move every branch whose last dependency this was from BLOCKED to READY.

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
  G9  a regression test exists AND fails on the base SHA
  G10 smoke passes after the merge, on the integration branch

REPORT
  ## MERGED                    yes/no
  ## NEW INTEGRATION SHA
  ## REBASE CONFLICTS          what, and how resolved
  ## FOCUSED TESTS POST-REBASE pasted
  ## SMOKE POST-MERGE          pasted (expect 457 passed)
  ## CHUNK D                   pasted, if applicable
  ## GATES                     G1..G10, one line each
  ## BRANCHES NOW UNBLOCKED
  ## ASSUMPTIONS INVALIDATED   for any still-pending branch
  ## LEDGER UPDATED            yes/no
```

---

## Rejection triggers

Do not merge; send to CORRECTION or REJECT:

- an earlier merge invalidated this branch's assumptions — rebase and rerun first
- the diff is correct but out of boundary
- the report claims a runtime behaviour with no pasted evidence
- a benchmark failure justifies code without an adjudication of
  `product_contract_violation` | `gold_data_defect` | `policy_disagreement`, citing the
  gold `relevance_note` / `export_rationale`
- the agent resolved an open question from `DECISIONS.md` § Open on its own authority

## On return from an overnight run

Do **not** merge everything that finished. Review in **dependency order**. Rebase each
branch onto the current integration commit and rerun its focused tests before applying
G1–G10. A branch whose assumptions an earlier merge invalidated is rejected, not patched
in place.
