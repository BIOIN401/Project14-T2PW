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
  G9. If this card CORRECTS or PRESERVES pre-existing observable behaviour, the
  proof MUST FAIL BEHAVIOURALLY at the base SHA and pass at the tip; SYMBOL
  ABSENCE IS NOT PROOF -- supply a shim or assert on artifact content. A
  genuinely NEW capability or module instead gets an EXPLICITLY LABELLED new
  acceptance test and needs no fabricated base failure. Mislabelling a regression
  as new functionality is a reject. Your reviewer verifies this at the base.

FOCUSED TESTS TO RUN
  <commands from TEST_MATRIX.md>

ACCEPTANCE — all must hold
  [ ] <checkable statement>
  [ ] <checkable statement>

TRAPS
  <paste only the relevant TRAP-n from [S7], verbatim>
```

---

## The card charter contains only this

Branch and **exact dispatch base** · **function-level ownership** · carried A0
requirements · **explicit exclusions** · hand-authored **and** generated budgets · focused
tests · applicable **real** G9 obligations · relevant traps · the **SBML prohibition** — no
SBML implementation, extension or refactor, and `src/t2pw/sbml/` is outside every
implementation boundary.

A charter **may be an external durable record**, as H-008's was, with its hashes reported
at closeout. **A tracked prompt commit is not required for every card.**

---

## Rules for whoever writes the prompt

- **One narrow, testable change per branch.** If the body needs more than about six
  `IMPLEMENT` steps, split the branch first. **Every card declares, before dispatch (D-019):**
  the **exact allowed manifest**; the **hand-authored** max additions-plus-deletions; and a
  **separate machine-generated-evidence budget** — max artifact count **and** a size limit,
  an explicit `0` when generation is unauthorized. Where **acceptance-criterion atomicity**
  constrains how small the card can be, state that rationale in the card. Any proposed split
  must name boundaries at which each half is **independently implementable and independently
  validatable** — never one that merges or leaves behind an unvalidated semantic half. A
  predicted or actual overrun obliges the implementer to **stop before committing** and obtain
  **renewed explicit authority with a revised budget**; see `[S4]`.
- **`OWNS` is function-level**, not file-level, wherever two branches share a file.
  `streamlit_app.py :: run_post_pipeline_sbml_artifacts` — never bare `streamlit_app.py`.
- **`MUST NOT CHANGE` is where correctness is preserved.** An agent that does not know a
  neighbouring function is already right will refactor it.
- **Never say "make test X pass."** Say what behaviour must hold. Otherwise TRAP-2 gets
  satisfied by reverting the fix.
- **Cite evidence, not conclusions.** "`_degree_zero_exports` at `:1876` runs after
  `_drop_quarantined_processes` at `:1862`" — not "there is an index bug."
- **Resolve bare filenames to canonical paths.** `MASTER_PLAN.md` § 9 has the table;
  `pipeline.py`, `map_ids.py` and `extract.py` all have re-export shims under `src/`
  that must never be edited.
- **Pure-move branches** (C-011, C-012) get a golden behavioural-equivalence test as
  their primary acceptance criterion, not a test count.
