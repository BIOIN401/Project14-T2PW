# docs/rag/ — RAG Subsystem Docs

Background + implementation briefs for adding Retrieval-Augmented Generation to
T2PW so it can generate **novel pathways from multiple papers**. Start with the
master plan: [`../rag_integration.md`](../rag_integration.md).

## Read in this order
1. [`00_overview.md`](00_overview.md) — goal, "novel pathway" definition, guardrails, glossary.
2. [`01_pipeline_placement.md`](01_pipeline_placement.md) — the R-stages and the exact seams.
3. [`02_vector_store.md`](02_vector_store.md) — store, schema, embeddings, config.
4. [`03_separation_invariant.md`](03_separation_invariant.md) — **the rule: do not bleed into the stages.**

## Implementation work packages (one sub-agent each)
`agents/` — dependency-ordered; WP0 first, WP7 last.
- [wp0_foundation.md](agents/wp0_foundation.md)
- [wp1_acquisition.md](agents/wp1_acquisition.md)
- [wp2_selection.md](agents/wp2_selection.md)
- [wp3_ingest_index.md](agents/wp3_ingest_index.md)
- [wp4_gap_retrieval.md](agents/wp4_gap_retrieval.md)
- [wp5_synthesis.md](agents/wp5_synthesis.md)
- [wp6_provenance_gates.md](agents/wp6_provenance_gates.md)
- [wp7_orchestration_ui.md](agents/wp7_orchestration_ui.md)

## Coordinator
[`MASTER_AGENT_PROMPT.md`](MASTER_AGENT_PROMPT.md) — prompt for the master agent that
deploys and verifies the WP sub-agents.
