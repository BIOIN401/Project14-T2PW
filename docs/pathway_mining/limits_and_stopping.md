# Limits and stopping

Every limit is configurable, recorded in `run_manifest.json`, and enforced by a
single `BudgetLedger`. A run always stops for a **named** reason.

## Defaults

Conservative on purpose: this is an exploratory feature and a runaway literature
crawl is both expensive and useless.

| Budget | Default | Env var (`PATHWAY_MINING_` +) | CLI flag |
|---|---|---|---|
| Max rounds | 8 | `MAX_ROUNDS` | `--max-rounds` |
| Max papers discovered | 150 | `MAX_PAPERS_DISCOVERED` | `--max-papers-discovered` |
| Max papers fully processed | 40 | `MAX_PAPERS_PROCESSED` | `--max-papers-processed` |
| Max new reactions per round | 25 | `MAX_NEW_REACTIONS_PER_ROUND` | — |
| Max passages per paper | 20 | `MAX_PASSAGES_PER_PAPER` | — |
| Max passages per extraction call | 5 | `MAX_PASSAGES_PER_CALL` | — |
| Max primary-graph reactions | 200 | `MAX_PRIMARY_REACTIONS` | — |
| Max candidate reactions | 300 | `MAX_CANDIDATE_REACTIONS` | — |
| Max candidate components | 40 | `MAX_CANDIDATE_COMPONENTS` | — |
| Max LLM calls | 200 | `MAX_LLM_CALLS` | `--max-llm-calls` |
| Max input tokens | 2,000,000 | `MAX_INPUT_TOKENS` | `--max-input-tokens` |
| Max output tokens | 400,000 | `MAX_OUTPUT_TOKENS` | `--max-output-tokens` |
| Max estimated cost (USD) | 5.00 | `MAX_COST_USD` | `--max-cost-usd` |
| Max wall clock (s) | 3600 | `MAX_WALL_CLOCK_SECONDS` | `--max-seconds` |
| Max frontier size | 60 | `MAX_FRONTIER_SIZE` | `--max-frontier-size` |
| Max results per literature query | 25 | `MAX_RESULTS_PER_QUERY` | — |
| Max query formulations per frontier item per round | 2 | `MAX_QUERY_ATTEMPTS` | — |
| Stop after N unproductive rounds | 2 | `STOP_AFTER_UNPRODUCTIVE_ROUNDS` | — |
| Input price per Mtok (USD) | 0.10 | `PRICE_INPUT_PER_MTOK_USD` | — |
| Output price per Mtok (USD) | 0.30 | `PRICE_OUTPUT_PER_MTOK_USD` | — |

The two prices exist so a cost budget can be enforced **before** any bill
exists. They are configured estimates and are labelled as such wherever the
number appears. They are not a billed amount.

Precedence: CLI flag / Streamlit control / config file > environment > default.

## Literature search per frontier item

Each frontier item's query **requires** the item's own terms and the target species.
The topic and generic words are not OR'd in. A bridge requires both of its sides. A reaction
label requires its first substrate and its first product. A trailing parenthetical note
(`HDA (16:2n6)` → `HDA`) is dropped from a term, because as a phrase search it almost never
matches. For example: `"12-OPDA" AND "Arabidopsis thaliana"`. An item with no usable term
(e.g. `? -> ?`) is exhausted without a query, not searched on the species alone.

* **Result window.** Each query asks `search_candidates` for `max_results_per_query` papers.
  Each configured source (EuropePMC, NCBI) is searched separately, and the lists are taken in
  turn before the cap. A single call over both sources caps the merged list, which let
  EuropePMC's first page fill it (pilot-001 F2). `max_papers_discovered` and
  `max_papers_processed` remain the hard bounds.
* **NCBI paper identity (IT2-2).** rag's PubMed parser takes the PMCID and DOI from any
  `ArticleIdList` in the record, reference list included, so the last cited reference's ids win.
  Pilot-002 recorded a 2021 paper as PMC411572 (a 1979 PNAS paper) and fetched two other articles'
  full texts that way. `rag/` is frozen, so mining keeps rag's own NCBI fetch (`_fetch_ncbi`) and
  its title, abstract and year. It re-reads the returned XML for each article's own
  `MedlineCitation/PMID` and never uses the parsed PMCID, DOI or URL. The PMCID is then asked of
  EuropePMC by PMID (`PaperSource._europepmc_item`, one `core` search per PMID per run); without
  a PMCID the PMID is the id, and the URL is the PubMed page. The same article found through
  EuropePMC and through NCBI therefore gets one `paper_id`, and is one paper. This path does not
  use rag's search cache for NCBI. The record's own `PublicationTypeList` is kept as its declared
  `article_type` (see [`evidence_policy.md`](evidence_policy.md) § evidence type).
* **A failed lookup is not "no PMCID" (IT2-N3).** EuropePMC answering with no record, or with
  no PMCID, is cached for the run. A lookup that failed (an exception, a non-200, an unreadable
  body) is not: the next search asks again. rag's `_resolve_europepmc_item` returns the same
  empty item for both, so one failure used to make an article a `PMID:` paper for the whole run,
  and the same article from EuropePMC a second, `PMC:` paper, two "independent" papers for
  `multiple_independent_papers`. An included paper's metadata lookup asks the same query again;
  when it now reveals the PMCID, the record keeps its `PMID:` id, gains its `pmcid`, and the
  `PMC:` id is marked seen, so EuropePMC's record of the article is the same paper.
* **Full-text check.** A fetched full text must open with the record's title: at least 70 % of
  the title's words (3+ letters, HTML unescaped) in its first 3,000 characters. Otherwise it is
  another article, it is refused, the abstract is used, and the paper's `notes` say so. Fixture
  and offline sources skip the check (`verify_full_text_title = False`).
* **Attempts.** When a query brings in no newly included paper, a second, distinct formulation
  is tried in the same round: the same required terms, plus the run topic. Each attempt is a
  row in `queries.jsonl` with its `attempt` number. An item is marked exhausted only after every
  attempt is tried without a new paper. It is marked exhausted at once when a query returns
  no result at all, because the second formulation only narrows the first.
* **A wider window instead of a second formulation.** When the topic is already one of the
  item's terms, there is no distinct second formulation. That is every `X | <topic>` bridge,
  which the frontier produces whenever the primary graph is empty, and the topic seed itself.
  If attempt 1 then filled its whole window (`result_count` equal to the papers asked for),
  attempt 2 asks the same query again for `min(2 × max_results_per_query,
  remaining max_papers_discovered)` papers. It is recorded as `attempt: 2`, with the same
  `query_text` and its own `query_id`. The item is exhausted only after that attempt too finds
  no new paper. A window that was not full exhausts the item after one attempt, because
  there is nothing past it.
* **A window the discovery budget cut.** A query asks for `min(max_results_per_query,
  remaining max_papers_discovered)` papers (a wider window, `min(2 × …, remaining)`). When the
  remaining budget makes that window smaller than it would otherwise be, the query still runs,
  but the item is **not** marked exhausted, whatever the query returned. A cut window only
  shows the top few hits, which are usually papers the run already has, so an empty result
  there says nothing about the item. That also holds when a wider window is due and the
  budget cannot fund it. In pilot-002, 21 of 25 exhausted items had only been asked through
  1-result windows. Once `max_papers_discovered` is reached, the round issues no further
  query (`check_papers_discovered`, before each item and each further attempt). The stop
  reason `max_papers_discovered_reached` is recorded, and the items left unasked or cut stay
  open on the frontier.

## Extraction in passage batches

A paper's passages are its own chunks, ranked against the frontier item's retrieval text
(A7). Retrieval for one paper filters the store to that paper before ranking; a store-wide top
`3 x k` filtered afterwards gave pilot-002's PMC5104190 "no passages retrieved" because other
papers' chunks outranked it. A paper is indexed with at most `max_passages_per_paper` chunks,
counted after its front matter, reference list, publisher notes and duplicate chunks are
removed ([`evidence_policy.md`](evidence_policy.md) § what text can be evidence); the chunks
past the cap are recorded as `over_passage_cap`.

A paper's retrieved passages (up to `max_passages_per_paper`) are extracted in consecutive
batches of `max_passages_per_call` (default 5). Each batch is its own logical model call, with
its own `model_calls.jsonl` record and `model_raw.jsonl` rows; its claims merge through the
claim store like any other paper's. A paper counts once toward `max_papers_processed`,
however many batches it takes. If the call budget runs out between batches, the rest of the
paper's passages are not sent, and a note on the paper says how many were.

Why: the extraction model reasons before it answers. Its hidden reasoning surveys every
passage and then drafts every claim with every per-field citation list. In pilot-002, 20
passages of one review needed well over 20,000 output tokens in a single call, and every
request came back empty on `finish_reason=length`. The output budget (`PATHWAY_MINING_MAX_TOKENS`,
12,000) is sized for one batch. See [`evidence/032-it2-extraction-diagnosis/`](evidence/032-it2-extraction-diagnosis/README.md).

## Stop reasons

`budget.STOP_REASONS` is a **closed set**; `BudgetLedger.stop` raises
`ValueError` on anything else, so a run cannot end for an unnamed reason.

| Reason | When |
|---|---|
| `max_rounds_reached` | round budget spent |
| `max_papers_discovered_reached` | discovery cap hit |
| `max_papers_processed_reached` | processing cap hit |
| `max_primary_reactions_reached` | primary graph full |
| `max_candidate_reactions_reached` | candidate quarantine full |
| `max_candidate_components_reached` | too many components |
| `max_new_reactions_per_round_reached` | the last round added at least that many reactions new to the graph (admitted or quarantined). That round is complete and kept; the next one does not start |
| `max_llm_calls_reached` | call budget spent |
| `max_input_tokens_reached` / `max_output_tokens_reached` | token budget spent |
| `max_estimated_cost_reached` | estimated cost budget spent |
| `max_wall_clock_reached` | time budget spent |
| `converged_no_useful_additions` | N consecutive rounds added nothing useful |
| `frontier_exhausted` | no unexhausted frontier items remain |
| `cancelled_by_user` | cooperative cancellation or `KeyboardInterrupt` |
| `preflight_failed` | a preflight check failed; no round ran |
| `unrecoverable_error` | an exception escaped a round |
| `completed_all_rounds` | the loop finished its work |

The **first** limit reached is the one recorded; a later one does not overwrite
it.

### What "useful" means

`GraphDelta.useful` is true when a round added a reaction, added a candidate
reaction, promoted one, updated one, or recorded a new conflict. A round that
only re-read papers it already had is unproductive even though it burned tokens.
Two of those in a row stop the run with `converged_no_useful_additions`. A
useful round resets the streak.

## Where limits are checked

* `check_before_round` — rounds, wall clock, calls, tokens, cost, papers
  processed, graph sizes, new reactions added by the last round, cancellation,
  convergence.
* `check_before_call` — calls, tokens, cost, wall clock, cancellation. Checked
  before **every** attempt, so a run that has exhausted its token allowance stops
  in `MiningModelClient` rather than inside the provider.
* `check_papers_discovered` — before each acquisition.
* `check_papers_processed` — before each frontier item **and before each paper is
  extracted**, together with the same cap per round. One query's result window
  (25 papers by default) can hold more papers than the cap allows; processing stops
  at the cap inside it, and the run stops with `max_papers_processed_reached`.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | completed normally |
| 2 | usage error (bad argument, missing file) |
| 3 | feature disabled |
| 4 | preflight failed |
| 5 | stopped without completing normally (limit, cancelled, failed) |

A limit-stop returns **5**, not 0, on purpose: it is a legitimate outcome, but a
caller scripting around it must be able to tell it from a run that finished its
work.

## Preflight

Run before anything expensive, and reported visibly. Checks: feature flag;
required inputs (species, and something to start from); output directory
writability; model configuration, credentials presence and — unless `--no-probe`
— reachability; embeddings configuration and reachability; acquisition sources;
the selected budgets; and the two standing policies (database validation
deferred, no PWML).

The model probe asks for one word with an output budget of 64 tokens
(`PATHWAY_MINING_PROBE_MAX_TOKENS` overrides it). A reasoning model can spend a
smaller budget thinking and return nothing with `finish_reason=length`; the check
then fails saying the probe budget was exhausted, not that the model is down.

A `fail` blocks the run: `completion_state` becomes `preflight_failed`,
`stop_reason` becomes `preflight_failed`, and **no model call is made**.

### The embeddings rule

`Embedder` degrades to deterministic lexical vectors when no embedding endpoint
answers. That is legitimate offline, but it must never be invisible. The
preflight reports which mode it found; `PassageIndex.embedding_mode` records
which mode the run was actually in; a degradation adds a warning to
`summary.json` and `report.md` saying it is an **infrastructure state, not a
finding about the literature**. Setting
`PATHWAY_MINING_REQUIRE_EMBEDDINGS=true` makes a missing embeddings service a
preflight **failure** instead of a warning.

> "RAG added nothing" is never an acceptable description of an embeddings outage.
