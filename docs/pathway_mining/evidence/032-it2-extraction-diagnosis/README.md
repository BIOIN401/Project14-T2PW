# ORCH-743 IT2-1: why pilot-002's two densest extraction calls came back empty

## What was replayed

Pilot-002's `mc_001_extract_claims_0002` (paper `PMC:PMC13364333`, the Marchantia jasmonate
review, 20 passages) was rebuilt from the read-only run
(`C:/t/mining-pilot-runs2/pilot-002-jasmonate-ath/`). Inputs were the same 20 spans in the order
sent (`model_calls.jsonl` `evidence_ids`), the same title, species and topic, and prompt
`mining-extract/1.1.0`. The run records only the last rung's `request_hash`. The rebuilt 35 %
rung hashes to that value, `sha256:b7b7fa136e3dcdcd` ([`dry_0002.json`](dry_0002.json)).
Every live call used the identical rung-1 user message, unless it says otherwise. The
script is [`diag_it2.py`](diag_it2.py). It calls `ensure_dotenv_loaded()` before any other
`t2pw` import.

`live` sends **one** provider request, with SDK retries off. The request is built with exactly
the kwargs `t2pw.llm.client.chat_detailed` sends: model, messages, temperature 0, `max_tokens`,
`response_format` json_schema and timeout. It uses that module's own configured client. It
can add `extra_body={"reasoning": ...}`, which is what the mining `reasoning` knob would have
to send (see below). Unlike `chat_detailed`, it also reads the provider name and
`usage.completion_tokens_details.reasoning_tokens`. Each call ran through `bounded_run.py`;
every `g11-*.json` shows exit `completed` and **FINAL SURVIVING COUNT 0**.

## The four live calls (budget: at most 4, at most $0.10)

| call | prompt | passages | `max_tokens` | extra | provider | finish | reasoning tok | content | est. cost |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.1.0 (as run) | 20 | 4,000 | `reasoning: {effort: "low"}` | Venice | `length` | 3,183 of 4,000 out | **0 chars** | $0.003511 |
| 2 | compact candidate "1.2.0" | 20 | 16,000 | none | DigitalOcean | `stop` | **0** | `{"claims": []}` (8 tokens) | $0.002289 |
| 3 | 1.1.0 (as run) | 20 | 16,000 | none | StreamLake | `length` | **16,001** of 16,000 out | **0 chars** | $0.007111 |
| 4 | 1.1.0 (as run) | 5 (the densest batch) | 12,000 | none | DigitalOcean | `stop` | **0** | 1 malformed claim, no citations (101 tokens) | $0.000677 |

Total estimated cost: **$0.013588**, at the mining config's $0.10 / $0.30 per Mtok. This is an
estimate, not a bill. Input tokens were 23,113 (1.1.0, 20 passages), 22,863 (compact prompt)
and 6,471 (batch 1).

## Findings

1. **The cause is output-budget exhaustion by hidden reasoning, and it scales with the
   work per call.** Call 3 spent all 16,000 output tokens reasoning (45,661 chars) and emitted
   nothing. Its reasoning splits into two parts:
   - The first 31,511 chars (≈ 69 %) survey the passages one by one.
   - The rest drafts the answer claim by claim, every per-field `*_evidence_ids` list
     included. "Reaction 3" through "Reaction 29" were drafted when the budget ran out.

   The final JSON would then restate all of it. For this review a single call needs well
   over 20,000 output tokens. Iteration 1's call 3 was also empty at 20,000
   (`013-item2-replay`). At the observed 41–83 tokens/s that is 4–8 minutes per request,
   against a 120 s request timeout.
2. **The `reasoning` knob is inert, and it would not have helped.**
   `MiningModelConfig.reasoning` is recorded on each `ModelCallRecord` and never sent.
   `modelclient._attempt` does not pass it. `t2pw.llm.client.chat_detailed` has no reasoning
   (or `extra_body`) parameter, and `llm/` is outside the mining boundary. Sent directly,
   `effort: "low"` did not prevent exhaustion at the branch's 4,000 tokens: call 1 is
   empty, like the pilot's 9 requests.
3. **Prompt compaction is not supported by the evidence.** The compact candidate
   (2,490 → 1,403 system chars, per-field rule kept) was served by a provider that did not
   reason at all. It answered `{"claims": []}` in 8 tokens for a review with ~30 reactions,
   so it neither proves nor disproves anything about reasoning length. The compact prompt is
   **not** adopted.
4. **Provider routing is not pinned, and one provider answers without reasoning and nearly
   empty.** Two of the four requests went to DigitalOcean. Both came back fast, with no
   reasoning and an answer that is wrong: an empty list for the whole review, and, for the
   densest batch, one claim whose "substrate" is `cis-to-iso isomerization`, with no
   citation. A `stop` status is therefore not evidence of a good extraction. Pinning or
   excluding a provider needs an `llm/client.py` change (`extra_body.provider`). That is out
   of scope here and is reported for the orchestrator.

## The fix (next commit)

The cause is too many passages per call for a reasoning model's output budget. The prompt,
the citation contract and the uncited-drop rule are unchanged. Two changes:

- **At most `max_passages_per_call` passages per extraction call** (new budget knob, default
  **5**). A paper's retrieved passages are extracted in consecutive batches, each its own
  logical call with its own lineage. Claims merge through the existing claim store.
- **An extraction output budget that fits a batch.** `MiningModelConfig.max_tokens` goes from
  4,000 to **12,000**, and `request_timeout_seconds` from 120 to **360** (12,000 tokens at
  the slowest observed 41 tokens/s is ≈ 293 s).

**Sizing, and what is NOT verified.** Batch 1 holds 126 of the 354 evidence-id mentions in
call 3's reasoning (36 %; the other batches have 34, 77 and 117). Scaled from call 3, a
5-passage batch needs about 4k tokens for the survey, 2–3k for the draft and 2–3k for the
answer, so roughly 9k, under 12k. The one batch call (call 4) went to the non-reasoning
provider, so **no reasoning provider has yet been observed answering a 5-passage batch
within 12,000 tokens**. The first pilot run after this change is that test.

## Files

| file | what |
|---|---|
| `diag_it2.py` | the replay script (dry / live / mining modes) |
| `dry_0002.json`, `g11-dry-0002.json` | the input-identity check |
| `callN_*_summary.json` | provider, finish reason, prompt / completion / reasoning tokens, content length, claims built by `build_claims`, estimated cost, wall time |
| `callN_*_raw.json` | the raw `content` and the provider's returned `reasoning` text, exactly as returned |
| `g11-live1..4.json` | the `bounded_run.py` report for each live call |

Checked before commit: no file here contains an OpenRouter key prefix (`sk-or-`) or any `sk-`
string.
