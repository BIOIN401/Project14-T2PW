# Implementation prompt template — `C-xxx`

Fill every `<...>`. Paste `[S1] [S3] [S4] [S5] [S6]` and the relevant `[S7]` traps
verbatim from `_SHARED_BLOCKS.md`. Do not summarize them.

---

```
[S1] [S3] [S4] [S5] [S6]

BRANCH        agent/<name>
BASE          <integration SHA at dispatch>   (cut from sprint/pwml-recovery)
WORKTREE      .claude/worktrees/<name>        (isolated; baseRef = head)
WAVE          <0|A0|A1|B|C|D|E>
DEPENDS ON    <merged branch IDs, or NONE>

OWNS — exclusive. A diff outside this list is an automatic reject.
  <path> :: <function or line range>
  <path> :: <function or line range>

OBJECTIVE
  <One paragraph. State the defect or gap with file:line evidence, then state
   what is true after the patch. Must end in a falsifiable sentence.>

MUST NOT CHANGE
  <Behaviours that must be byte-identical afterwards. Name the functions and
   why they are already correct, so the agent does not "improve" them.>

IMPLEMENT
  1. <step>
  2. <step>

TESTS YOU MUST ADD
  <name> :: <the exact failure it catches>
  Every regression test MUST FAIL on the base SHA. Your reviewer will verify
  this by checking out the base, applying only your test, and running it.

FOCUSED TESTS TO RUN
  <commands from TEST_MATRIX.md>

ACCEPTANCE — all must hold
  [ ] <checkable statement>
  [ ] <checkable statement>

TRAPS
  <paste only the relevant TRAP-n from [S7], verbatim>
```

---

## Rules for whoever writes the prompt

- **One narrow, testable change per branch.** If the body needs more than about six
  `IMPLEMENT` steps or the estimate exceeds ~400 lines, split the branch first.
- **`OWNS` is function-level**, not file-level, wherever two branches share a file.
  `streamlit_app.py :: run_post_pipeline_sbml_artifacts` — never bare `streamlit_app.py`.
- **`MUST NOT CHANGE` is where correctness is preserved.** An agent that does not know a
  neighbouring function is already right will refactor it.
- **Never say "make test X pass."** Say what behaviour must hold. Otherwise TRAP-2 gets
  satisfied by reverting the fix.
- **Cite evidence, not conclusions.** "`_degree_zero_exports` at `:1876` runs after
  `_drop_quarantined_processes` at `:1868`" — not "there is an index bug."
- **Pure-move branches** (C-011, C-012) get a golden behavioural-equivalence test as
  their primary acceptance criterion, not a test count.
