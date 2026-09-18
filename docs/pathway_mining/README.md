# Pathway Mining (experimental)

An **experimental, default-off** literature-mining mode for T2PW. Given a species
and one or more starting points, it iteratively mines the scientific literature
and builds an **auditable research graph** of paper-supported biochemical
reactions.

> ## Two statements that govern everything below
>
> 1. **This mode produces a research graph, NOT PWML.** The PWML exporter is
>    never invoked, PathWhiz identifiers are never required, importer-only layout
>    rules are never applied, and strict PWML gates never run. A research-graph
>    failure is never reported as a PWML failure. See
>    [`no_pwml.md`](no_pwml.md).
> 2. **External biochemical database validation is DEFERRED in version 1.**
>    PathBank, PWC, HMDB, KEGG, ChEBI, PubChem and UniProt are not consulted, are
>    not a prerequisite for admitting a reaction, and are not queried during
>    mining rounds. Every entity and reaction carries
>    `{"status": "not_evaluated"}`. Admission is on **paper evidence alone**. See
>    [`evidence_policy.md`](evidence_policy.md).

It is additive: with `T2PW_PATHWAY_MINING_ENABLED` unset, **no existing command,
default, import, output, cache or PWML behaviour changes**, and importing
`t2pw.mining` does no work beyond defining names.

---

## The questions it answers

* What biochemical network can the literature support?
* How far can the system expand from a topic, seed, paper set, or existing pathway?
* Which papers and passages support every reaction?
* In which mining round was each entity or reaction discovered?
* Which portions form a chemically connected pathway?
* Which relevant findings remain disconnected candidate subgraphs?
* Where does the literature agree, conflict, or leave uncertainty?

---

## Quick start

```bash
# 1. enable (or pass --enable-pathway-mining per invocation)
export T2PW_PATHWAY_MINING_ENABLED=true

# 2. check services BEFORE spending anything
python -m t2pw.mining preflight --species "Arabidopsis thaliana" --topic "jasmonate biosynthesis"

# 3. run
python -m t2pw.mining run \
  --species "Arabidopsis thaliana" \
  --topic "jasmonate biosynthesis" \
  --seed-metabolite "12-OPDA" \
  --paper PMID:12345678 \
  --max-rounds 8 \
  --max-cost-usd 10

# 4. explore
python -m t2pw.mining list
python -m t2pw.mining status runs_mining/<run_id>
streamlit run src/t2pw/mining/ui/app.py
```

---

## Data flow

```
              inputs                     (topic - species - seeds - papers - existing graph)
                |
                v
        [ frontier items ]  <---------------------------------------+
                |                                                    |
                v                                                    |
        literature queries    (traceable; enzyme/pathway/gene names   |
                |              are SEARCH TERMS, never connections)   |
                v                                                    |
             papers           acquire -> screen -> cache -> ingest    |
                |                                                    |
                v                                                    |
            passages          retrieve (vector + lexical)             |
                |                                                    |
                v                                                    |
             claims           extract; EVERY FIELD cites a passage    |
                |                                                    |
                v                                                    |
        evidence decisions    accepted | candidate | conflicting |    |
                |             incomplete | rejected | hypothesis      |
                v                                                    |
        research graph        primary graph  ||  candidate subgraphs  |
                |             (chemistry only) (quarantined, kept)    |
                |                                                    |
                +----> graph delta ----> next frontier ---------------+
                                             (round N+1)
```

The loop stops at the first limit reached, or after two consecutive rounds that
add nothing useful, and **always records which**.

---

## The distinction the whole design rests on

**Paper / provenance edges and biochemical connectivity are different
relationships, and they are drawn in different views.**

| | Reaction network view | Paper / evidence provenance view |
|---|---|---|
| Edge meaning | chemical: substrate into a reaction, reaction out to a product | bibliographic: paper -> evidence span -> claim -> reaction |
| What an edge proves | these two reactions share a metabolite in compatible roles | this paper supports this claim |
| What it does **not** prove | nothing about who wrote it | **nothing about chemistry** |

Two reactions mentioned in the same paper are **not** connected. Two reactions
catalysed by the same enzyme are **not** connected. Only a shared *chemical
participant* in compatible roles connects anything, and
`t2pw.mining.graph.connection_between` — the single function that can answer the
question — is not given enzymes, papers, pathway names or text to look at.

---

## Documentation map

| Document | Covers |
|---|---|
| [`architecture.md`](architecture.md) | modules, seams into the existing RAG stack, round anatomy |
| [`feature_flag.md`](feature_flag.md) | the three activation paths, what "off" guarantees |
| [`streamlit.md`](streamlit.md) | the Pathway Mining page and the main-app hook |
| [`cli.md`](cli.md) | every command, flag and exit code |
| [`evidence_policy.md`](evidence_policy.md) | what admits a claim; the deferred database layer |
| [`species_policy.md`](species_policy.md) | the seven species states and the ortholog rule |
| [`graph_semantics.md`](graph_semantics.md) | primary vs candidate, bridges, promotion |
| [`visualizations.md`](visualizations.md) | the two research views, their filters and guarantees |
| [`artifacts.md`](artifacts.md) | the run directory and every file's schema |
| [`limits_and_stopping.md`](limits_and_stopping.md) | budgets, stop reasons, defaults |
| [`resume_and_cancellation.md`](resume_and_cancellation.md) | checkpoints, resuming, cancelling |
| [`no_pwml.md`](no_pwml.md) | why no PWML, and what a later conversion layer would need |
| [`known_limitations.md`](known_limitations.md) | what this version does badly or not at all |
| [`testing.md`](testing.md) | the fixture corpus and how to run the suite |

---

## Status

Experimental research feature on branch `experiment/pathway-mining`. **Not
merged anywhere.** Production PWML engineering remains frozen under `D-090`; this
work is separate and does not alter production behaviour. See
[`known_limitations.md`](known_limitations.md) for what is deliberately deferred.
