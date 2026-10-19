# CLAUDE.md

Project instructions for T2PW (Text-to-PathWhiz) live in **`AGENT_INSTRUCTIONS.md`**
(pipeline architecture, key source files, stage contracts, conventions). Read that first
for anything outside the sprint below.

---

## PWML Recovery Sprint

An active multi-agent sprint runs on `sprint/pwml-recovery`, cut from `research-mode`
@ `9e1b9abe7ba8a1a228558fd03ca6c394cc22c31e`.

**Read before doing any sprint work:**

| Document | Why |
|---|---|
| [`docs/pwml_recovery_sprint/PRODUCT_CONTRACT.md`](docs/pwml_recovery_sprint/PRODUCT_CONTRACT.md) | What "correct" means. **Outranks any test, benchmark result, or inference from the code.** |
| [`docs/pwml_recovery_sprint/MASTER_PLAN.md`](docs/pwml_recovery_sprint/MASTER_PLAN.md) | Waves, dependencies, branch register, file-ownership map, standing traps |
| [`docs/pwml_recovery_sprint/LEDGER.md`](docs/pwml_recovery_sprint/LEDGER.md) | Live task state. The single source of truth for what is dispatched, merged, blocked |
| [`docs/pwml_recovery_sprint/DECISIONS.md`](docs/pwml_recovery_sprint/DECISIONS.md) | Locked product decisions. Append-only, product owner only |
| [`docs/pwml_recovery_sprint/TEST_MATRIX.md`](docs/pwml_recovery_sprint/TEST_MATRIX.md) | Test chunks, runtimes, per-branch obligations, benchmark milestones |

### Roles

The main Claude session is the **Lead Orchestrator and Integration Authority**. It
dispatches, sequences, routes reviews, and merges. **It does not implement coding patches
and never approves its own work.**

Subagents: `pwml-implementer` (isolated worktree, one branch) · `pwml-reviewer` (no edit
tools) · `pwml-test-runner` (fixes nothing) · `pwml-bio-auditor` (read-only adjudication).

### Permanent merge rules

No patch merges unless **all** of these hold:

1. Its dependency is already merged.
2. Its diff stays within the assigned file/function boundary.
3. Focused tests pass.
4. Existing affected tests pass — or a pinned baseline moved deliberately, with an exact
   documented delta.
5. An independent reviewer approved the **actual diff**, not the report.
6. It does not weaken a biological gate to increase PWML production.
7. It preserves incomplete-but-correct pathways as `review_required` rather than dropping
   them.
8. It does not allow exporters to repair biology after the canonical graph is frozen.
9. **G9.** A claimed correction or preservation of pre-existing observable behaviour
   carries a proof that **fails behaviourally on the base SHA** and passes at the tip;
   **symbol absence is not proof**. A genuinely new capability or module instead carries an
   **explicitly labelled new acceptance test** and needs no fabricated base failure.
   Mislabelling a regression as new functionality is a reject.
10. The integration smoke suite (**508** tests, ~40 s) passes after the merge.
    465 was stale from `C-054`; `C-067` moved it to 473, `C-106` to 503, and 508 is the
    number `D-098` merge gate 10 and `ORCH-724` actually measured. `TEST_MATRIX.md`
    still says 503 in three places; corrected in place there under `ORCH-736`.
11. **Test-process lifecycle holds.** Every test, benchmark, pipeline leg and LLM-backed
    command ran through the bounded foreground wrapper, and the cleanup report shows
    **zero surviving owned processes**. A run with survivors is an *infrastructure
    failure*, not a test result. See `TEST_MATRIX.md` § 0.

**A benchmark failure does not by itself justify a code change.** Classify it first as
`product_contract_violation`, `gold_data_defect`, or `policy_disagreement`, citing the
gold `relevance_note` / `export_rationale`. Only the first justifies code.

### Logging is not optional

**Every piece of work is written into its doc before the task is reported done.** Work that
exists only in a chat transcript is lost: the transcript is not searchable by the next
session, is not reviewable, and cannot be cited in a merge gate. A task reported complete
with nothing written down is not complete.

| what you did | where it goes |
|---|---|
| a task, start to finish | `docs/pwml_recovery_sprint/ORCH-<id>-RESULT.md` — one per task; the full record, including what failed and why |
| its live state: dispatched, merged, blocked | a row in `LEDGER.md` (append-only) |
| a defect, a measurement, a wrong theory you ruled out | a numbered finding in the task's RESULT doc, and `FINDINGS.md` when it outlives the task |
| a product decision | `DECISIONS.md` — **product owner only.** Never append to it yourself; record the open question in the RESULT doc instead |
| a test or benchmark obligation that changed | `TEST_MATRIX.md`, in place, with the exact delta |
| a bounded-run report, proof or verdict | `evidence/g11/<TASK-ID>/` — committed, never pasted (gate G11 rule 9) |
| evaluation artifacts: gold, variants, run outputs | under `docs/evaluation_v2/`, with a README in the directory |

Three rules about what the entry has to contain:

1. **Record the failures and the dead ends, not just the result.** A theory you tested and
   disproved is worth more than a conclusion, because it stops the next session re-running
   it. If you were wrong, write down what you believed and what disproved it.
2. **Cite the artifact, never the claim.** A number in a doc names the file it came from —
   a wrapper report, a proof, a payload path. "Tests pass" with no report path is not a
   record, and under G11 it is not evidence either.
3. **State what is NOT settled.** Open questions go in the RESULT doc under a heading the
   product owner can act on, with the options and what each costs. Silence reads as
   agreement.

Docs-only commits are made separately from code, and `git status --porcelain` is checked
afterwards to confirm nothing else moved.

### Sprint-wide constraints

- **Every test, benchmark and pipeline command runs through the bounded foreground
  wrapper.** No detached processes, no `nohup`, no untracked background jobs. Cleanup
  targets only PIDs the job created — `taskkill /IM python.exe` and `pkill python` are
  forbidden. One heavy job at a time; never `pytest -n auto`. `TEST_MATRIX.md` § 0.
- **`--basetemp=<dir>` on every pytest invocation.** Without it 83 tests error with
  `PermissionError` and you will report a false regression. Never run the full suite
  unchunked (~16 GB).
- **Never commit a cache modification.** `data/enrichment_cache.json` is 39 MB and
  tracked; `.git` is already 158 MB.
- **Do not rebuild what exists.** See `MASTER_PLAN.md` § 2 — the UniProt accession fetch,
  the provenance carrier, alias traversal, the RAG gap detector, the RAG admission gate
  and several others are already implemented and tested.
- Identical legs give materially different Stage-1 draws at temperature 0. Re-run a leg
  before calling a single-leg change a regression.
