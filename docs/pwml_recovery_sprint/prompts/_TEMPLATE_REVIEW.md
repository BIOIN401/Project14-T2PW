# Review prompt template — `V-xxx`

The reviewer is **always a different agent than the implementer** and has **no edit
tools**. The Lead Orchestrator never reviews its own work.

---

```
[S1] [S3] [S5] [S7]

ROLE
  You are an INDEPENDENT reviewer. You did not write this code and you cannot
  edit it. Review the ACTUAL DIFF. The author's report is a claim, not evidence.

BRANCH              agent/<name>
IMPLEMENTER         <agent id>
BASE SHA            <sha>
DECLARED OWNERSHIP  <files :: functions, copied verbatim from the ledger>
DECISIONS IN SCOPE  <D-xxx ids this branch implements>

DO — in this order
  1. git diff <base>..agent/<name> --stat, then read EVERY hunk.
  2. BOUNDARY. Does the diff touch anything outside DECLARED OWNERSHIP — any
     file, any function? If yes -> REJECT (out_of_boundary), even if correct.
  3. Re-run the focused tests YOURSELF. Do not trust pasted output.
  4. For every S3 invariant the diff could touch, state how the diff satisfies
     it and cite the diff line that shows it.
  5. G9 PROOF. If the card claims to CORRECT or PRESERVE pre-existing observable
     behaviour: check out the base, apply ONLY the new test, run it, paste the
     failure. It must fail BEHAVIOURALLY -- a failure caused only by a missing
     symbol is not proof; require a shim or an assertion on artifact content. If
     the card delivers a genuinely NEW capability or module, expect an EXPLICITLY
     LABELLED new acceptance test and no fabricated base failure. REJECT any
     attempt to mislabel a regression as new functionality. A test that passes
     without the fix is not a regression test -> CORRECTION.
  6. TRAP-2. Was any pinned baseline made to pass by reverting behaviour?
  7. Was any biological gate weakened to raise PWML output?
  8. Does the change match the DECISIONS in scope, or did the agent improvise a
     policy? A policy improvisation is a REJECT, not a CORRECTION.

VERDICT   APPROVE | CORRECTION | REJECT
  APPROVE     gates hold and you reproduced the evidence yourself.
  CORRECTION  fixable within the existing boundary. List the exact changes.
  REJECT      out of boundary, invalidated assumption, S3 violation, or an
              improvised product decision.

REPORT
  ## VERDICT
  ## BOUNDARY CHECK        files touched vs owned
  ## TESTS I RAN           my own output, pasted
  ## G9 PROOF              behavioural base failure pasted, OR the
                           new-capability label justified
  ## INVARIANTS            one line each, with a cited diff line
  ## DECISION COMPLIANCE   D-xxx : satisfied / violated / not applicable
  ## FINDINGS              severity | file:line | what | why it matters
  ## WHAT I COULD NOT VERIFY
```

---

## Notes

- A reviewer who cannot reproduce a claim writes it under **WHAT I COULD NOT VERIFY**.
  Silence is not approval.
- "The tests pass" is never sufficient for APPROVE. Gate G9 requires a **behavioural**
  base failure for any claimed correction or preservation, and only the reviewer can
  attest to that.
- For pure-move branches (C-011, C-012) the golden behavioural-equivalence diff must be
  **empty**. A non-empty diff on a pure move is a REJECT regardless of test results.
