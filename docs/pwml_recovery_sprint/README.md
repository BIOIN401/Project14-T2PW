# PWML Recovery Sprint — Control Plane

This directory is the sprint's control plane. It holds no implementation code.

| File | What it is | Who writes it |
|---|---|---|
| `PRODUCT_CONTRACT.md` | The biological product requirements and the non-negotiable release behaviour. The authority when a benchmark result and the code disagree. | Product owner only |
| `MASTER_PLAN.md` | Dependency graph, waves, branch register, file-ownership map, merge gates, standing traps. | Lead Orchestrator |
| `LEDGER.md` | Live per-task state. The single source of truth for what is dispatched, merged, blocked. | Lead Orchestrator only |
| `TEST_MATRIX.md` | Test chunks, runtimes, which branches owe which suites, benchmark milestones. | Lead Orchestrator |
| `DECISIONS.md` | Locked product decisions with dates. Append-only. | Product owner |
| `BASELINE.md` | The pre-sprint measurement every acceptance claim is compared against. | Filled by INIT-001 |
| `prompts/` | Prompt templates plus expanded per-task prompts. | Lead Orchestrator |
| `evidence/` | Committed artifacts backing sprint claims: golden fixtures, benchmark JSON, before/after diffs. | Agents, via their branch |

## Roles

**Lead Orchestrator / Integration Authority** — the main Claude session. Dispatches,
sequences, routes reviews, merges, gates. **Writes no implementation code and never
approves its own work.**

**Implementer** (`.claude/agents/pwml-implementer.md`) — one narrow branch, isolated
worktree, exclusive file ownership.

**Reviewer** (`.claude/agents/pwml-reviewer.md`) — reviews the actual diff. Has no
edit tools by construction.

**Test runner** (`.claude/agents/pwml-test-runner.md`) — runs suites and benchmarks,
reports numbers, fixes nothing.

**Biological auditor** (`.claude/agents/pwml-bio-auditor.md`) — adjudicates whether a
benchmark failure is a product-contract violation, a gold-data defect, or a policy
disagreement. Read-only.

## Reading order for a new agent

1. `PRODUCT_CONTRACT.md` — what "correct" means here.
2. Your task prompt in `prompts/`.
3. `MASTER_PLAN.md` § your branch's register row.
4. Nothing else. Do not read other branches' prompts.

## Sprint status

Initialization complete. **Wave A0 not yet dispatched — awaiting product-owner
approval of the setup report.**
