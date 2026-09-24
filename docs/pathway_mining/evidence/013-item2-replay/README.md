# ORCH-743 item 2: why pilot-001 dropped 59 of 59 enzyme, gene and species values

## What was replayed

Pilot-001's only extraction call (`mc_001_extract_claims_0001`, paper `PMC:PMC13364333`)
was rebuilt from the read-only pilot record (`C:/t/mining-pilot/docs/pathway_mining/pilot/001/run/`).
It used the same 20 spans, in the order the pilot sent them (`model_calls.jsonl` `evidence_ids`),
and the same paper title, species and topic. It ran through `extract_claims`, with item 1's raw
lineage in place. The script is [`replay_extraction.py`](replay_extraction.py). It calls
`t2pw.config.ensure_dotenv_loaded()` before any other `t2pw` import, because the mining CLI
does not load `.env`.

**Identical input, proven before any spend.** In `dry` mode no model is called. The rebuilt
system and user messages hash to `sha256:9c0d58077db82d98`, which is the pilot's own
`request_hash` ([`dry.json`](dry.json)).

## The three live calls (the budget was at most 3)

Each call is one request: `max_attempts=1`, `LLM_MAX_RETRIES=1`, `deepseek/deepseek-v4-flash`,
temperature 0. Every call ran through `bounded_run.py`. Each `g11-*.json` shows exit
`completed` and **FINAL SURVIVING COUNT 0**.

| call | prompt | output budget | result | tokens in / out | est. cost |
|---|---|---|---|---|---|
| 1 | pilot's, byte-identical | 4,000 (the branch default) | `empty_completion`, `finish_reason=length`, 0 chars | 22,843 / 4,000 | $0.003484 |
| 2 | pilot's, byte-identical | 16,000 | `truncated_completion`, `finish_reason=length`, 9,565 chars | 22,843 / 16,000 | $0.007084 |
| 3 | fixed (`mining-extract/1.1.0`) | 20,000 | `empty_completion`, `finish_reason=length`, 0 chars | 23,113 / 20,000 | $0.008311 |

Total estimated cost: **$0.018879**. This is the configured price estimate, not a bill.

## What the raw output shows

Call 2's raw completion ([`call2_pilot_prompt_16k_raw.json`](call2_pilot_prompt_16k_raw.json)) was
truncated after 9 complete claims. Those 9 claims fill `enzyme`, `gene` and `species` 9 times each.
Every participant carries `evidence_ids` pointing at shown spans, and every claim carries a
claim-level `evidence_ids`. **The completion contains no per-field `*_evidence_ids` key at all:
0 of 27.** Key census:

```
evidence_ids 37 · name 29 · substrates 10 · products 10 · enzyme 9 · gene 9 · species 9 ·
species_inferred 9 · hypothesis 9 · claims 1
```

So the values were not cited under another key or in another format, and the parser did not
misread anything. **The model never wrote the per-field citation lists**, and the old
`build_claims` correctly dropped every uncited value.

Why the model omitted them:

* The schema does not require them. It requires only `substrates`, `products` and `evidence_ids`.
* The prompt names them only as a wildcard ("Each field has an *_evidence_ids list").
* The model probably never saw the schema. It wrote exactly the keys the prompt text names
  (`enzyme`, `species`, `species_inferred`, `hypothesis`, `substrates`, `products`,
  `evidence_ids`). It never wrote a key that appears only in the schema (`direction`,
  `compartment`, `reversible`, `conditions`, `tissue`, `pathway_names`, `confidence`,
  `entity_type`, `stoichiometry`), although several would apply to these desaturase steps. That
  is consistent with the provider not applying `response_format: json_schema` for this model. It
  is an inference from the output, not something measured.

## The fix, and what each part is proven by

1. **Prompt** (`mining-extract/1.1.0`). Rule 1 now names each per-field list. Rule 7 shows the
   exact JSON shape, with each `*_evidence_ids` beside its field. **Not verified live.** Call 3,
   the verification call, spent all 20,000 output tokens reasoning and returned no content (see
   the risk below). No call was left to retry it.
2. **Deterministic fallback** in `extraction.build_claims`. It covers an enzyme, gene or species
   value whose own `*_evidence_ids` list is **missing**, which is the failure seen here. Such a value
   is kept only if it occurs verbatim in one of that claim's own cited participant spans. It is
   attributed to that span and recorded as `verbatim_cited_span`. An explicit empty list, or one that
   cites only unshown ids, is respected and the value is dropped. That keeps the test fixture's
   ambiguous-species case (P6, `species_evidence_ids: []`) held back, as before. **Verified offline on the real
   model output from call 2** ([`fallback_on_raw.py`](fallback_on_raw.py) →
   [`call2_fallback_offline.json`](call2_fallback_offline.json), no model call). Of 27 asserted
   values, **23 are kept** as `verbatim_cited_span` and **4 are still dropped**: `Mp2g00660` and
   `Marchantia polymorpha` on the two MpLOX1 claims, whose cited participant spans do not contain
   them. Every kept species is *Marchantia polymorpha*. The species guardrail still admits none of
   them to an *A. thaliana* graph: 7 are labelled `inferred_only` because the cited evidence
   contains "Ortholog" table headers (pilot finding F3, in `species.py`, out of scope) and the
   2 MpLOX1 claims are labelled `in_vitro_or_cell_free`.

## Risk found on the way (out of scope, not fixed)

This model spends most of its output budget reasoning. At the branch default of 4,000 tokens, one
of two draws with the pilot's exact input returned nothing (call 1). At 16,000 it was still
truncated (call 2). At 20,000 with the longer prompt it again returned nothing (call 3). Pilot-001's
single draw fit in 3,629 tokens. A pilot-002 extraction call can therefore come back empty. In
that case the existing shrink ladder retries with 60 % and then 35 % of the passages, which costs
calls and loses passages. `MiningModelConfig.max_tokens` and the provider's reasoning controls are
outside ORCH-743's four items and were not changed.

## Files

| file | what |
|---|---|
| `replay_extraction.py`, `fallback_on_raw.py` | the two scripts, run from `C:/t/mining-iter/` |
| `dry.json` | the request-hash identity check |
| `callN_*_raw.json` | item 1's `model_raw` rows for each call: the raw completion text exactly as returned |
| `callN_*_summary.json` | status, the `ModelCallRecord`, claims and dropped fields |
| `call2_fallback_offline.json` | `build_claims` (with the fallback) over call 2's complete claims |
| `g11-*.json` | the `bounded_run.py` report for each of the five runs |

Checked before commit: no file here contains an OpenRouter key prefix.
