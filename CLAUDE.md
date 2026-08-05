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
9. It adds a regression test for the demonstrated failure, and that test **fails on the
   base SHA**.
10. The integration smoke suite (457 tests, ~40 s) passes after the merge.

**A benchmark failure does not by itself justify a code change.** Classify it first as
`product_contract_violation`, `gold_data_defect`, or `policy_disagreement`, citing the
gold `relevance_note` / `export_rationale`. Only the first justifies code.

### Sprint-wide constraints

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
