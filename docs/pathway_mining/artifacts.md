# Run artifacts

Every run writes one self-contained directory. Nothing is needed from outside it
to read, audit or regenerate the report.

```
runs_mining/<run_id>/
    run_manifest.json          config.json            preflight.json
    papers.jsonl               queries.jsonl          frontier.jsonl
    evidence_spans.jsonl       claims.jsonl           decisions.jsonl
    model_calls.jsonl          costs.json             conflicts.json
    primary_graph.json         candidate_subgraphs.json
    summary.json               report.md
    checkpoint.json            state.json
    graph_views/
        reaction_network.json      reaction_network.dot
        provenance_network.json    provenance_network.dot
        evidence_table.json
    rounds/
        round_001/ round.json checkpoint.json input_frontier.jsonl queries.jsonl
                   papers_considered.jsonl passages_retrieved.jsonl
                   claims_extracted.jsonl decisions.jsonl rejections.jsonl
                   new_frontier.jsonl graph_delta.json metrics.json errors.jsonl
        round_002/ ...
```

`run_id` is `<UTC timestamp>_<8-char digest of species+topic>`.

## Two rules

* **Flush per round.** Every top-level artifact is rewritten at the end of each
  round, so a run that is cancelled or killed keeps every completed round.
* **Rewrite, never append.** Each snapshot is written from in-memory state, so a
  resumed run cannot double-write a row and a half-written append is never
  mistaken for data.

## Schema versioning

| Constant | Value | Bumped when |
|---|---|---|
| `ARTIFACT_SCHEMA_VERSION` | `t2pw.mining/1.0.0` | a persisted field changes meaning |
| `PROMPT_VERSION` | `mining-extract/1.0.0` | an extraction prompt changes in a way that could change a claim |
| `VIEW_SCHEMA_VERSION` | `t2pw.mining.view/1.0.0` | a view payload's shape changes |

The manifest carries the first two; every claim carries both; every checkpoint
and both view payloads carry their own.

---

## `run_manifest.json`

```json
{
  "run_id": "2026-09-17_142233_9f1b2c3d",
  "schema_version": "t2pw.mining/1.0.0",
  "prompt_version": "mining-extract/1.0.0",
  "started_at": "...Z", "updated_at": "...Z", "finished_at": "...Z",
  "completion_state": "stopped_at_limit",
  "completed_normally": false,
  "stop_reason": "converged_no_useful_additions",
  "stop_detail": "2 consecutive rounds added nothing useful",
  "git": {"commit": "...", "branch": "experiment/pathway-mining",
          "dirty": false, "dirty_file_count": 0},
  "environment": {"python": "3.13.6", "platform": "...", "packages": {...}},
  "feature_flags": {"T2PW_PATHWAY_MINING_ENABLED": "true",
                    "enabled_via": "cli_flag", "offline": false},
  "config": { ...the full resolved MiningConfig: species, topic, starting inputs,
              budgets, model config, sources, backend, runs dir... },
  "rounds_completed": 4,
  "resumed_from_round": -1,
  "notes": [],
  "produces_pwml": false,
  "external_database_validation": "deferred_not_evaluated"
}
```

`completion_state` is one of `running`, `completed`, `stopped_at_limit`,
`cancelled`, `failed`, `preflight_failed`. Everything except `completed` is in
`INCOMPLETE_STATES` and adds a note saying the run did not complete normally.
**A cancelled or interrupted run never reads as completed.**

## The JSONL files

One JSON object per line; safe to stream.

| File | One row per | Key fields |
|---|---|---|
| `papers.jsonl` | paper (included **and** excluded) | ids, metadata, screening outcome, inclusion/exclusion reasons, evidence type, discovery round + query |
| `queries.jsonl` | literature query | query text, frontier item + kind, round, sources, source terms, result count, new paper ids, cache hit, error |
| `frontier.jsonl` | open frontier item | kind, target, reason, priority, origin round, attempts, exhausted |
| `evidence_spans.jsonl` | quoted passage | exact text, section, page, paragraph, locator, chunk, offsets, round, query, frontier item, score |
| `claims.jsonl` | extracted claim | everything in [`evidence_policy.md`](evidence_policy.md) § claims, including `field_evidence` |
| `decisions.jsonl` | admission decision | status, reason, round, target, evidence level, species state, `supersedes` |
| `model_calls.jsonl` | model call | requested vs **actual** model, provider, prompt/schema version, reasoning, input/output/cached tokens, estimated cost, latency, retries, finish reason, parse and schema errors, fallback used + reason, request/response hashes, attempt log |

## `costs.json`

```json
{
  "estimate_only": true,
  "note": "Estimated from configured per-million-token rates. This is NOT a billed
           amount and does not reflect provider discounts, cached-token pricing or
           minimum charges.",
  "price_input_per_mtok_usd": 0.10, "price_output_per_mtok_usd": 0.30,
  "llm_calls": 9, "input_tokens": 12345, "output_tokens": 2345,
  "cached_tokens": 0, "estimated_cost_usd": 0.001938,
  "max_cost_usd": 5.0, "elapsed_seconds": 41.2
}
```

The estimate is labelled as such everywhere it appears, including in `report.md`.

## `primary_graph.json` / `candidate_subgraphs.json`

The primary file carries the selected component, its selection rationale, the
cofactor-bridge setting, the reaction nodes, the chemical connections among them,
and the graph hash. The candidate file carries each quarantined component with
its `quarantine_reason`, `created_round`, `promoted_round`, `relevance_score`,
`rationale`, `promotion_history`, `bridge_reaction_ids` and its full reaction
records.

## `conflicts.json`

Every `ConflictRecord`: attribute, claim identity, each position with its claim
ids, paper ids and evidence ids, first/last round, `resolved`, and a
`resolution_note` of `cross_paper` or `same_paper`.

## `graph_views/`

Two payloads, one table, two Graphviz sources. See
[`visualizations.md`](visualizations.md).
`provenance_network.json` carries
`"provenance_edges_are_not_chemical_connectivity": true` and states it in its
`note`; the DOT source prints the same caveat in its graph label.

## Per-round directories

`rounds/round_NNN/round.json` holds the whole round; the same content is also
split into per-concern files for convenience. Each round records: input
frontier, generated queries, papers considered, passages retrieved, claims
extracted, admission decisions, rejections with reasons, graph delta (with
before/after graph hashes), the newly generated frontier, round metrics, errors
and retries, and the checkpoint.

`metrics` carries counts plus the cumulative `usage` snapshot and the primary /
candidate / component sizes at the end of that round.

## `checkpoint.json` and `state.json`

`checkpoint.json` (also copied into the round dir) is the resumable cursor: round
index, completion flag, graph hash, the open frontier, processed and discovered
paper ids, seen query ids, unproductive streak and cumulative usage.

`state.json` is the full resumable state: the graph, the claim store (claims,
identities, histories, conflicts), papers **and their text**, evidence spans,
queries, frontier, model calls, deltas and errors. Carrying the paper text is
what lets a resumed run rebuild its passage index without re-fetching — and
without a second round of charges.

## `summary.json` and `report.md`

`summary.json` is every number the report prints, computed once: papers
discovered / screened / included / excluded / processed and by evidence type;
passages; claims extracted, distinct identities, latest status counts, rejections
by reason; graph sizes, connections, multi-paper reactions, species-state census,
promoted components; conflicts; additions by round; usage and costs; model-call
statuses and failures; retrieval stats including the **embedding mode**; the
preflight report; warnings; errors; and provenance problems.

`report.md` renders that for a human, leading with a banner when the run did not
complete normally, and stating both standing policies (no PWML, database
validation deferred) in the first screen.

## Regenerating

```bash
python -m t2pw.mining report runs_mining/<run_id>
```

rebuilds `summary.json`, `report.md`, both view payloads, both DOT files and the
evidence table **from the artifacts alone** — no model call, no network, no
re-run. It works on a cancelled run.

## Observability

Logs are structured records (`ModelCallRecord`, round `metrics`, `GraphDelta`)
with stable ids, round indices, latencies and token counts, so an exporter to
Phoenix or OpenTelemetry could be attached later by walking `model_calls.jsonl`
and the round directories. **No external observability platform is a requirement
for version 1**, and none is contacted.

## Secrets

None are written. The manifest records model *slugs* and whether an API key was
*present*. A test walks every file of a finished run asserting that none of
`api_key`, `sk-`, `OPENROUTER_API_KEY` or `password` appears anywhere.
