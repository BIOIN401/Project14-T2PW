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
                   new_frontier.jsonl frontier_cut.jsonl graph_delta.json
                   metrics.json errors.jsonl model_raw.jsonl
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
| `ARTIFACT_SCHEMA_VERSION` | `t2pw.mining/1.1.0` | a persisted field changes meaning |
| `PROMPT_VERSION` | `mining-extract/1.1.0` | an extraction prompt changes in a way that could change a claim |
| `VIEW_SCHEMA_VERSION` | `t2pw.mining.view/1.0.0` | a view payload's shape changes |

The manifest carries the first two; every claim carries both; every checkpoint
and both view payloads carry their own.

---

## `run_manifest.json`

```json
{
  "run_id": "2026-09-17_142233_9f1b2c3d",
  "schema_version": "t2pw.mining/1.1.0",
  "prompt_version": "mining-extract/1.1.0",
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
| `papers.jsonl` | paper (included **and** excluded) | ids, metadata, screening outcome (`mining_rescue` when the rescue re-admitted it; the production reason is then in `notes`), inclusion/exclusion reasons, evidence type and the rule that decided it (`evidence_type_basis`), whether the declared type was checked against EuropePMC (`evidence_type_verified`), declared `article_type`, discovery round + query; `notes` say what indexing kept out of the paper (front matter, reference list, ...) |
| `queries.jsonl` | literature query | query text, frontier item + kind, round, attempt (formulation 1 or 2), sources, source terms, result count, new paper ids, cache hit, error |
| `frontier.jsonl` | open frontier item | kind, target, reason, priority, origin round, attempts, exhausted, `relevance` (the on-route score, 0-3, of a `missing_species_evidence` item; 0 otherwise) |
| `evidence_spans.jsonl` | quoted passage | exact text, section, page, paragraph, locator, chunk, offsets, round, query, frontier item, score |
| `claims.jsonl` | extracted claim | everything in [`evidence_policy.md`](evidence_policy.md) § claims, including `field_evidence` |
| `decisions.jsonl` | admission decision | status, reason, round, target, evidence level, species state, `supersedes` |
| `model_calls.jsonl` | model call | requested vs **actual** model, provider (`not_reported_by_llm_client` when the shared client says nothing, which it does today; `""` only for a call that made no request), prompt/schema version, reasoning, input/output/cached tokens, estimated cost, latency, retries, finish reason, parse and schema errors, fallback used + reason, request/response hashes, attempt log (every provider request of every rung, each tagged with its `rung` and `shrink_fraction`), `answer_quality` (`ok`, `retried_after_thin`, `thin_after_retry`, `thin_not_retried`; `""` when no answer was accepted; see [`evidence_policy.md`](evidence_policy.md) § A thin answer). For an accepted answer, `finish_reason`, `response_hash`, `response_chars` and `provider` describe the answer kept; when a failed re-ask leaves the earlier thin answer kept, the re-ask's parse or schema error is not on the record, only on its `model_raw.jsonl` row |

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
before/after graph hashes), the newly generated frontier, the items the frontier
cap cut, round metrics, errors and retries, and the checkpoint.

`metrics` carries counts plus the cumulative `usage` snapshot and the primary /
candidate / component sizes at the end of that round, and `frontier_cut`: the
`count` of items the cap cut and their `by_kind` counts.

`frontier_cut.jsonl` has one row per item the frontier cap dropped in that round
(`frontier_item_id`, `kind`, `target`, `priority`, `relevance`), and is empty when
nothing was cut. See [`limits_and_stopping.md`](limits_and_stopping.md) § The
frontier cap and the species reserve.

### `model_raw.jsonl` — raw completion lineage

One row per **attempt** of every extraction call made in that round (a call that
climbs the shrink-and-retry ladder has one row per rung, and the one re-ask of a thin
answer is a row of its own). `model_calls.jsonl`
keeps only a `response_hash`; this file keeps what the model actually said, so a
dropped field or a failed parse can be attributed to the completion itself.

| Field | |
|---|---|
| `call_id` | links the row to its `model_calls.jsonl` record |
| `round_index` · `purpose` · `paper_id` | as on the call record |
| `call_status` | the call's final status (`ok`, `parse_error`, `truncated`, ...) |
| `answer_quality` | the call's `answer_quality`, as on its record |
| `attempt` · `model` | ladder rung (1-based) and the slug asked |
| `shrink_fraction` · `passages_sent` · `chars_sent` · `schema` | what the rung sent: the fraction of the user message kept (1.0, 0.6, 0.35), how many passages it held (a passage cut mid-text counts), its length, and `full` / `tight` / `none` |
| `provider` | the provider the client reported for this attempt, or `not_reported_by_llm_client` |
| `provider_requests` | every provider request the shared client made for this rung (it re-sends an empty 200), from its diagnostics `attempt_log` |
| `finish_reason` | as the provider reported it for that attempt |
| `parse_outcome` | `ok`, `thin_answer`, `parse_error`, `truncated_completion`, `truncated_but_parseable`, `schema_error`, `empty_completion` or `provider_error` |
| `thin_reason` · `retry_of_thin_answer` | why a `thin_answer` is thin; `true` on the row of the one re-ask it caused |
| `input_tokens` · `output_tokens` · `latency_ms` | this attempt's own usage (the client's token counter, else the 4-chars-per-token estimate), so a thin answer is measurable. It is per attempt, not per provider request: the client's counter cannot split one attempt's re-sent empty replies |
| `error` | the parse / schema / provider error, if any (a key-like `sk-…` token is redacted) |
| `response_hash` | the provider diagnostics' hash of that attempt's text |
| `raw_text` | the completion text exactly as returned |

Only model **output** is written: never the request, its headers or any key. The
same rows are also inside that round's `round.json`.

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
evidence-level census (`graph.evidence_levels`), promoted components; conflicts;
additions by round; usage and costs; model-call statuses and failures; retrieval stats including the **embedding mode** and what
indexing kept out (`excluded`, `excluded_by_paper`: front matter, reference lists,
publisher notes, duplicate and over-cap chunks, each with a count and characters); the
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
