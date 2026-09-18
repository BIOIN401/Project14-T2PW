# CLI

```
python -m t2pw.mining [--enable-pathway-mining] <command> [options]
```

The CLI calls the same `MiningController` and artifact layer the Streamlit page
calls. There is no parallel implementation; a test asserts that neither front
end contains a round loop.

`--enable-pathway-mining` goes **before** the subcommand and enables the feature
for that invocation only.

## Commands

### `run` — start a mining run

```bash
python -m t2pw.mining run \
  --enable-pathway-mining \
  --species "Arabidopsis thaliana" \
  --topic "jasmonate biosynthesis" \
  --seed-metabolite "12-OPDA" \
  --seed-enzyme "AOS" \
  --paper PMID:12345678 \
  --paper PMC7654321 \
  --paper-file ./notes/preprint.pdf \
  --starting-graph ./existing_pathway.json \
  --max-rounds 8 \
  --max-cost-usd 10
```

| Flag | Meaning |
|---|---|
| `--species` | **required.** The target species. |
| `--topic` | Topic to mine. |
| `--seed-metabolite` / `--seed-enzyme` / `--seed-reaction` | Repeatable seed entities. |
| `--paper` | Repeatable. `PMID:123`, `PMC123`, a DOI, or a supported URL. |
| `--paper-file` | Repeatable. A local PDF or text file to ingest. |
| `--starting-graph` | An existing pathway / research-graph JSON, used **read-only**. |
| `--config` | A JSON file of overrides (`{"budgets": {...}, "model": {...}}`). |
| `--run-id`, `--runs-dir` | Where the run directory goes. |
| `--model` | Model slug. Default `deepseek/deepseek-v4-flash`. |
| `--offline` | Never contact a literature service. |
| `--no-screen` | Skip the eligibility screen. |
| `--vector-backend` | `memory` (default) or `chroma`. |
| `--max-rounds`, `--max-papers-discovered`, `--max-papers-processed`, `--max-llm-calls`, `--max-input-tokens`, `--max-output-tokens`, `--max-cost-usd`, `--max-seconds`, `--max-frontier-size` | Budgets. See [`limits_and_stopping.md`](limits_and_stopping.md). |
| `--no-probe` | Skip the live preflight probes (still runs the offline checks). |
| `--json` | Print the summary as JSON instead of the human table. |

All starting-input kinds compose: a topic, a species, two seed metabolites,
several PMIDs and an existing pathway can be supplied together.

Progress prints one line per round:

```
[round 2] starting, frontier=17
[round 2] papers=4 claims=3 primary=3 candidates=2 useful=yes
[run] stopped_at_limit (converged_no_useful_additions)
```

### `resume` — continue an interrupted run

```bash
python -m t2pw.mining resume runs_mining/<run_id> --max-rounds 12
```

Restores the graph, claims, decisions, papers, evidence and frontier from
`state.json` + `checkpoint.json`, re-indexes the papers whose text survived (no
re-fetch, no second round of charges), and continues in the **same** run
directory. Token, call and cost budgets stay cumulative; the wall clock
restarts. See [`resume_and_cancellation.md`](resume_and_cancellation.md).

### `status` — inspect a run

```bash
python -m t2pw.mining status runs_mining/<run_id> [--json]
```

```
run_id            2026-09-17_142233_9f1b2c3d
completion_state  stopped_at_limit  <- did NOT complete normally
stop_reason       converged_no_useful_additions
rounds_completed  4
papers            9 included / 0 excluded / 8 processed
primary graph     5 reactions, 12 chemical connections
candidates        2 reactions in 2 components
conflicts         1 unresolved
model calls       9  est. $0.001938 (estimate, not a bill)
PWML              none - this mode produces a research graph only
database check    deferred (not_evaluated)
```

Exits **5** when the run did not complete normally, **0** when it did.

### `report` — regenerate from artifacts

```bash
python -m t2pw.mining report runs_mining/<run_id>
```

Rebuilds `summary.json`, `report.md`, both view payloads, both Graphviz sources
and the evidence table from what is on disk. No model call, no network, no
re-run. Works on a cancelled run.

### `preflight` — check services without running anything

```bash
python -m t2pw.mining preflight --enable-pathway-mining \
  --species "Arabidopsis thaliana" --topic "jasmonate biosynthesis" [--no-probe] [--json]
```

Exits **0** if every check passes, **4** otherwise.

### `list` — list runs

```bash
python -m t2pw.mining list [--runs-dir DIR] [--json]
```

## Cancellation

`Ctrl-C` during a run is caught: the ledger is cancelled, the current round's
artifacts are flushed, the manifest records `cancelled` / `cancelled_by_user`,
and the process exits **5**. The run is resumable.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | ok |
| 2 | usage error |
| 3 | feature disabled |
| 4 | preflight failed |
| 5 | run did not complete normally |
