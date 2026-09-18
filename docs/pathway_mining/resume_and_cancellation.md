# Resume and cancellation

## Cancellation is cooperative and honest

Three ways to ask a run to stop:

* `Ctrl-C` in the CLI (caught as `KeyboardInterrupt`);
* `controller.cancel()` from code or a UI callback;
* any budget being exhausted (a stop, not a cancellation).

A cancellation takes effect at the next round or call boundary. The current
round's artifacts are flushed first, then:

```json
"completion_state": "cancelled",
"completed_normally": false,
"stop_reason": "cancelled_by_user",
"notes": ["this run did NOT complete normally (state=cancelled); every completed
          round's artifacts are present and nothing is presented as final"]
```

`report.md` leads with a banner saying the same thing. **A cancelled or
interrupted run is never reported as a completed one.** The CLI exits 5.

Every completed round keeps its directory, its checkpoint and its metrics. A run
cancelled after round 2 has `rounds/round_001/` and `rounds/round_002/`, both
with `checkpoint.json`, and the top-level artifacts reflect the state at the end
of round 2.

## Checkpoints

Written at the end of every round, to both
`rounds/round_NNN/checkpoint.json` and the top-level `checkpoint.json`:

| Field | |
|---|---|
| `round_index`, `completed`, `written_at` | the cursor |
| `graph_hash` | content hash of placement, for comparing runs |
| `frontier` | the open frontier items |
| `processed_paper_ids`, `discovered_paper_ids` | what not to redo |
| `seen_query_ids` | which questions were already asked |
| `unproductive_streak` | so convergence survives a resume |
| `usage` | cumulative rounds, calls, tokens, cost, papers, passages, claims |
| `schema_version` | |

Alongside it, `state.json` holds the full resumable state: the graph, the claim
store (claims, identities, decision histories, conflicts), the papers **and
their text**, the evidence spans, the queries, the frontier, the model-call
records, the deltas and the errors.

Carrying the paper text is deliberate: it lets a resumed run rebuild its passage
index without re-fetching anything, so resuming costs no acquisition calls and no
re-ingestion.

## Resuming

```bash
python -m t2pw.mining resume runs_mining/<run_id> [--max-rounds N] [--max-cost-usd X]
```

or in Streamlit, the **Resume an interrupted run** expander.

`MiningController.resume` rebuilds the config from `config.json`, the state from
`state.json`, the cursor from `checkpoint.json`, re-indexes the surviving paper
text, and continues **in the same run directory** — the round numbering
continues, earlier round directories are untouched, and the manifest records:

```json
"resumed_from_round": 2,
"notes": ["resumed from round 2 at 2026-09-17T15:04:11+00:00"]
```

### Budget semantics on resume

* **Cumulative:** rounds, LLM calls, input tokens, output tokens, estimated cost,
  papers discovered, papers processed, passages, claims, unproductive streak.
  A resumed run cannot spend the budget twice.
* **Fresh:** the wall clock. The elapsed time of a run interrupted yesterday is
  not a resource this process is consuming, so it restarts.

`--max-rounds` and `--max-cost-usd` on the resume command raise (or lower) the
budget for the continuation; the new value is recorded.

### Setup is not repeated

A resumed controller has `_setup_done = True`: the starting graph is not
re-adopted, explicit papers are not re-loaded, and every already-known paper id
is pre-seeded into the acquirer, so nothing is fetched twice.

### The preflight on resume

`continue_run()` defaults `probe_model=False` and `probe_embeddings=False`: the
offline checks still run (flag, inputs, output directory, sources, budgets,
policies) but no live probe is spent again.

## Determinism

Ids are content-addressed (`ids.py` uses no counter, no clock and no UUID), so a
resumed run mints the same id for the same thing as the run it is resuming, and
`graph_hash` is comparable across runs and across a resume.

Two caveats on end-to-end determinism:

* the model is not deterministic even at temperature 0 (identical legs give
  materially different draws — a known property of this stack), so two runs over
  the same corpus can differ;
* the literature services return what they return on the day.

The *policy* is deterministic: given the same claims, the same admission
decisions, the same placement and the same primary-component selection follow,
and a test asserts the selection is stable across repeated builds.
