# PF-746-02 — llm/client.py: provider routing unpinned and unreported, reasoning unsent, partial response_hash, cumulative-only usage

Status: REGISTERED, not dispatched. Production is frozen (D-102); dispatch needs a product-owner decision.

## Defect

All line numbers are in `src/t2pw/llm/client.py` at `b4bb0b74`.

1. **Provider routing is not pinned, and the provider is not reported.** The `chat_detailed` request is built at :653-669 from `model`, `messages`, `temperature`, `max_tokens`, `timeout` and an optional `response_format`. It sends no OpenRouter `provider` preference, and the file has no `extra_body` anywhere. `chat_with_tools._call_once` builds its own request at :878-893, also with none. `CompletionDiagnostics` (:361-370) has the fields `model, stage, attempts, finish_reason, response_status, terminal_reason, request_hash, response_hash, raw_chars, attempt_log`. None of them is a provider field, and nothing in the module reads a provider from the response (`resp`).
2. **No reasoning setting is ever sent.** The signature of `chat_detailed` (:603-613) has no reasoning parameter, and the request at :653-659 carries none.
3. **`response_hash` is partial.** It is assigned in one place only, the success path: `diagnostics.response_hash = _hash(text)` (:767). `text` is the **stripped** reply of the final, successful attempt (:759). `known_limitations.md:166-167` on 82fc69a7 calls it "the hash of the last request of a call", which is not what the code does. Three gaps follow:
   - The content-filter exit (:722) and the empty-after-retries exit (:757) return with `response_hash == ""`.
   - `CompletionDiagnostics.note` builds `attempt_log` rows (:395-404) with only `attempt, status, content_chars, finish_reason?, error?`, so no row carries a hash.
   - `note` publishes the row through `_publish_attempt` (:412 → :340-341, `response_hash=diagnostics.response_hash`) *before* :767 assigns the hash. The leg-trace value is therefore always `""`, even on success. The success path calls `diagnostics.note(...)` at :761 and only then runs `diagnostics.response_hash = _hash(text)` at :767.

   A fix must also allow for this: `_hash(None)` returns `""` (:290-291), so an empty reply has to be hashed as `""`, not as `None`.
4. **Usage exists only as a process-wide cumulative counter.** `_record_usage` (:118-124) adds into the module dict `_token_stats` (:111), and `get_token_stats` (:127-138) returns totals. No usage is attached to `CompletionDiagnostics` or to any `attempt_log` row, so the tokens of one provider request cannot be recovered.

## Observed impact

- **Production leg traces (observed).** The eval-v2 sample report committed at b4bb0b74 counted 663 `LEG_TRACE` `model_attempt` rows from 20 leg traces in `runs_verify/2026-09-01_1612` (`SAMPLE_REPORT.md:73`). `response_hash` is present on 0/663, `provider_backend_identity` on 0/663 and `token_usage` on 0/663 (`docs/evaluation_v2/sample_report.json:18320-18372`; `SAMPLE_REPORT.md:81, 85, 86`). The report names the same note-before-assign ordering as the cause of the empty hash (`src/t2pw/bench/eval_v2/repro.py:56-62`). This card would move only the `response_hash` count. The other two need `leg_trace.py` fields, which are out of scope here.
- **Pilot-003 provider (observed).** `provider` is `""` on all 16 call records, and `actual_model` only echoes the requested slug (`PILOT-003-AUDIT.md:10, 566`, on `experiment/pathway-mining-pilot-003`). Pilot-003's mining client also never filled `ModelCallRecord.provider` (:566). With no provider in `CompletionDiagnostics`, it had nothing to fill it from.
- **Pilot-003 thin answers (observed count, inferred cause).** The reviewer's probe found 7 of the 16 first-attempt answers structurally thin: no cited field, or a schema placeholder used as a value. All seven share one alphabetically sorted key set (`docs/pathway_mining/evidence/orch745/reviews/csc_r0.txt:44, 67` on 82fc69a7).
  - By call total, three used 154-209 output tokens and four used 5k-24k.
  - `_0008`'s total of 24,122 includes two empty 12k requests; its answering request used about 122 (`PILOT-003-AUDIT.md:548`).
  - **The cause is inferred.** The reviewer inferred from the shared key shape that the provider route or decoder produced these answers (`csc_r0.txt:48`). The pilot-003 audit inferred "a provider or mode that did not reason" for four of them from token counts and latency (`PILOT-003-AUDIT.md:575`). No provider was recorded, so neither inference can be checked against these artifacts (`known_limitations.md:153-158`).
- **Empty replies have no hash (observed).** Pilot-002's empty-completion row and the 013 replay record both have `"response_hash": ""` (`PILOT-002-AUDIT.md:360`; `docs/pathway_mining/evidence/013-item2-replay/call1_pilot_prompt_summary.json:35` on 82fc69a7).
- **Reasoning (observed that nothing was sent; the effect is inferred).** Pilot-002's config had `reasoning: ""`, so no reasoning budget or effort was passed. Two calls each billed 36,000 output tokens over 9 requests, all ending `finish_reason=length` with empty content (`PILOT-002-AUDIT.md:347-348, 366`). The audit attributes this to reasoning using up `max_tokens` (:365-366). That attribution is itself inferred, because only total output tokens are reported. **Whether sending a reasoning setting would have prevented it has not been tested.**
- **Per-request usage was unavailable (observed).** The pilot-003 auditor derived `_0008`'s answering-request output (122) from the call total of 24,122. It subtracted 2 x 12,000 for the two `length` requests, which assumes each one used exactly `max_tokens` (`PILOT-003-AUDIT.md:537, 548`).

## Current mitigation in mining

These are on 82fc69a7, in `src/t2pw/mining/modelclient.py`. They mitigate the problem; they do not fix it.
- **Thin answers.** `thin_answer` (:190-215) detects a thin answer by its structure. When the budget check allows it (:425-426), `call_json` re-asks once and sets `answer_quality` to `ok`, `retried_after_thin`, `thin_after_retry` or `thin_not_retried` (:80-83, :423-449). It cannot choose the route. The reviewer estimated that re-asks add about 687 s to a 1081 s round-1 model total (`csc_r0.txt:67`).
- **Provider.** When nothing is reported it falls back to `PROVIDER_NOT_REPORTED = "not_reported_by_llm_client"` (:76-78, :539-544). It already reads `diagnostics.provider`, or a `provider` key on an `attempt_log` row, if either is present (:541-543).
- **Reasoning.** It is recorded only (`record.reasoning` at :341; `config.py:174, 201`, a `str`). The `chat_detailed` call (:504-513) does not pass it.
- **Usage.** Each attempt's tokens are the difference between `get_token_stats()` before and after the call, read through the `_token_stats` wrapper (:289-298, :496, :553-555). If both differences are 0, it falls back to an estimate of about 4 characters per token (`_estimate_tokens`, :303-305, used at :556-558). This gives per-attempt (per-rung) totals, not per-request ones.
- **`response_hash`.** It is copied from the diagnostics as-is (:519, :575). There is no mitigation for the empty case.

## Proposed change

All changes are in `src/t2pw/llm/client.py`. With nothing configured, the request stays exactly as it is today.
1. **Routing.** Read one module setting once at import, for example `OPENROUTER_PROVIDER_ORDER` (a comma list), next to `REQUEST_TIMEOUT_SECONDS` (:47-52). When it is set and `PROVIDER == "openrouter"`, add `extra_body={"provider": {"order": [...], "allow_fallbacks": <product-owner choice>}}` in both request builders, `chat_detailed` and `chat_with_tools._call_once`. When it is unset, add nothing.
2. **Reporting.** Add `provider: str = ""` to `CompletionDiagnostics` and to `to_dict`. In `chat_detailed`, read `getattr(resp, "provider", None)` after each `create()`. `repro.py:41` names this hook. Confirm the field name against one authorised live response before merging.
3. **Reasoning.** Add an optional `reasoning: Optional[Dict[str, Any]] = None` parameter to `chat_detailed`. Send it as `extra_body["reasoning"]` only when it is not None and `PROVIDER == "openrouter"`. The signature of `chat()` is unchanged, so `chat()` does not take `reasoning`. It still gets the routing setting, because it delegates to `chat_detailed` (:591).
4. **Hashes and usage per attempt.** On every attempt that got a response (ok, empty, content_filter), pass these to `note()`:
   - `response_hash=_hash(<the stripped text the caller receives>)`. For an empty reply that text is `""`, so the value is `_hash("")`, never `_hash(None)`.
   - `provider`.
   - `prompt_tokens` and `completion_tokens` from `resp.usage`.
   - `usage_reported` (bool).

   `note()` adds each of these keys to the row only when the caller passes it, so a bare `note()` row keeps today's shape. Set `diagnostics.response_hash` on the content-filter and empty exits as well. In `_publish_attempt`, pass the row's own `response_hash`. Keep the set of `record_model_attempt` keyword arguments unchanged. Leave `_token_stats`, `get_token_stats` and `reset_token_stats` unchanged.

Not in this card: publishing the provider and usage to `LEG_TRACE` (that needs `leg_trace.py`), `request_hash` covering only the messages (`repro.py:47-50`), and any mining-side change. Item 4 is independent of items 1-3. It is the only part with a behavioural correction, and it can be dispatched alone.

## Ownership

- `src/t2pw/llm/client.py` :: the module settings block next to `REQUEST_TIMEOUT_SECONDS` (:47-52), for the one new routing setting only
- `src/t2pw/llm/client.py` :: `CompletionDiagnostics` (fields :361-370, `to_dict` :372-384, `note` :386-412)
- `src/t2pw/llm/client.py` :: `_publish_attempt` (:332-342), the `response_hash` argument only
- `src/t2pw/llm/client.py` :: `chat_detailed` (signature :603-613, request :653-669, response paths :684-769)
- `src/t2pw/llm/client.py` :: `chat_with_tools._call_once` (request :878-893), the routing `extra_body` only
- new file `tests/test_llm_client_provenance.py`

## Acceptance

1. With the routing setting unset and `reasoning=None`, the arguments recorded by a stub `create()` are identical to b4bb0b74's for the same inputs, in both `chat_detailed` and `chat_with_tools`. There is no `extra_body` key.
2. With the routing setting set and `PROVIDER == "openrouter"`, both request builders send `extra_body["provider"]`. With `PROVIDER == "local"`, neither sends `extra_body`.
3. Under openrouter, `chat_detailed(..., reasoning={"effort": "high"})` sends `extra_body["reasoning"] == {"effort": "high"}`.
4. Every `attempt_log` row from an ok, empty or content_filter attempt carries these five keys:
   - `response_hash == _hash(stripped text)`
   - `provider`
   - `prompt_tokens`
   - `completion_tokens`
   - `usage_reported`

   `to_dict()` includes `provider`. A row from a bare `note(attempt=..., status=..., content_chars=...)` call is unchanged.
5. After an empty-after-retries exit or a content-filter exit, `to_dict()["response_hash"] == _hash("")`.
6. The `response_hash` that `_publish_attempt` passes for an attempt with a response equals that row's hash. The set of `record_model_attempt` keyword arguments is unchanged.
7. For the same stubbed responses, the following are identical to b4bb0b74:
   - the returned `text`
   - the number of attempts
   - backoff sleeps
   - `terminal_reason`
   - `get_token_stats()` totals
8. These files pass unmodified:
   - `tests/test_llm_client_empty_completion_retry.py`
   - `tests/test_llm_client_request_timeout.py`
   - `tests/test_c111_timeout_observability.py` (it pins the exact bare-`note()` row at :392-396)
   - `tests/test_extraction_escalation_ladder.py`
   - `tests/test_preprocessor.py`
9. SMOKE (508) passes, run through the bounded wrapper with zero surviving processes.
10. No test makes a network or LLM call.

## G9 obligation

Reuse the pattern from `tests/test_llm_client_empty_completion_retry.py`:
- the `openai` stub (:55-70);
- the `SimpleNamespace` response builder `_resp` (:92-100), extended with `usage` and `provider` where needed;
- the recording fake client (:103-124) installed over `client_mod._client`;
- the fixture (:127-150), which replaces the module-level `client_mod.time` with a sleep recorder (:141-143). It does not patch `time.sleep`.

**Corrections (each must fail behaviourally at b4bb0b74):**
- **(a) Exit hashes.**
  - Set `LLM_MAX_RETRIES=2` and have the stub return `""` twice with finish_reason `stop`. Assert `to_dict()["response_hash"] == _hash("")`, and that each `attempt_log` row's `response_hash` equals it.
  - Repeat with a single `""` reply whose finish_reason is `content_filter`.
  - At base the diagnostics value is `""` and the rows have no `response_hash` key, so both tests fail.
- **(b) Published hash.** Monkeypatch `t2pw.batch.leg_trace.record_model_attempt` with a capturing function. This works because `_publish_attempt` imports it at call time (:328). Have the stub return `"abc"`, then assert the captured `response_hash == _hash("abc")`. At base the captured value is `""`, because `note()` publishes at :761 before :767 assigns the hash. This is the mechanism behind eval-v2's 0/663.

**New capability (labelled NEW ACCEPTANCE, so no base failure is claimed):**
- **(c)** Given a response with `provider="X"`, `diagnostics.provider` and the row's `provider` are `"X"`. A response with no `provider` attribute gives `""`.
- **(d)** Recorded request arguments for routing set and unset, and for reasoning set and unset (acceptance items 1-3). The unset halves are preservation tests and also pass at base.
- **(e)** An empty reply with `usage(10, 5)` followed by an ok reply with `usage(7, 3)` gives per-row tokens of 10/5 and 7/3. `usage=None` gives `usage_reported=False`. The `get_token_stats()` deltas are unchanged, and that part also passes at base.

## Risks and interactions

- **Production callers.**
  - `chat_detailed`: `pipeline.py:3737`, `preprocessor.py:185`, `localized_repair.py:1018-1020` (lazy import, then the call).
  - `chat()` (:571-600) delegates to `chat_detailed`. All 11 modules under `src/t2pw` that import `chat` (for example `pipeline.py`, `gap_resolver.py`, `map_ids.py`) would therefore also get the routing setting.
  - `chat_with_tools`: `gap_resolver.py:2733`, `pathway_curator.py:316`.
  - `get_token_stats`: `streamlit_app.py:5689`, a frozen, protected file that must not be touched.
  - `src/llm_client.py:1` re-exports the module with `*`.
- **The pipeline prefers the client hash.** `pipeline.py:3802, 3855, 4107` use `call_diag["response_hash"] or payload_hash(raw)`, where `raw` is the returned (stripped) `completion.text`. For a string, `payload_hash` (`extraction_diagnostics.py:270-293`) computes the same value as `_hash` (:281-301). If the client hashes the same stripped text, the values the extraction ladder feeds to `identical_empty_hash` (`extraction_ladder.py:432-435`) do not move. Hashing the raw, unstripped content would change them for any reply with leading or trailing whitespace, including whitespace-only replies.
- **The new keys are persisted.** The pipeline stores `attempt_log` and the whole `to_dict()` in boundary records (`pipeline.py:3810, 3859`). `_bounded_attempt` (`extraction_diagnostics.py:828-849`) passes unknown keys through. The preprocessor also keeps `to_dict()` (`preprocessor.py:194`). The new row keys and `provider` will appear in those artifacts.
- **The leg trace drops unknown arguments silently.** `LegTrace.model_attempt` accepts a fixed set of keyword arguments (`leg_trace.py:313-325`). An unknown one raises a `TypeError` that is swallowed at :386-389, and the whole row is lost without any error. Do not add arguments to `record_model_attempt` in this card.
- **Pinning trades availability.** With `allow_fallbacks` false, an outage at the pinned provider would cost retries and then a failure. Whether it surfaces as `APIError` retries ending in a `RuntimeError` (:812-815, :831) or as an immediate `BadRequestError` (:777-798) depends on OpenRouter's response, which has not been verified. Whether pinning removes thin answers has not been tested.
- **Mining.**
  - With no mining change, `modelclient.py:541-543` would start reporting the provider.
  - Mining copies every `attempt_log` row into `record.attempt_log` (:521-532), so the new row keys would appear in mining records.
  - The thin-answer re-ask stays.
  - Mining's token differencing would be a separate, mining-side change. So would passing `reasoning`, which mining configures as a `str` (`config.py:174`) and the proposal takes as a dict.
- **Tests.** The two client test files and `test_c111_timeout_observability.py` are not in the SMOKE list (`TEST_MATRIX.md:259-271`), so they must run as focused tests. SMOKE does include tests that import the client: `test_empty_extraction_payload.py` and `test_completeness_audit.py`.
- **Live verification is not authorised here.** Checking the response's `provider` field name, and the reasoning acceptance, needs one billed call.

## References

- `src/t2pw/llm/client.py` @ b4bb0b74 (lines above)
- @ b4bb0b74:
  - `src/t2pw/bench/eval_v2/repro.py:21-67`
  - `docs/evaluation_v2/sample_report.json:18320-18372`
  - `docs/evaluation_v2/SAMPLE_REPORT.md:73-86`
  - `src/t2pw/batch/leg_trace.py:313-345, 375-389`
  - `src/t2pw/pipeline/pipeline.py:3737-3810, 3847-3859, 4100-4110`
  - `src/t2pw/pipeline/extraction_diagnostics.py:270-293, 828-849`
  - `src/t2pw/pipeline/extraction_ladder.py:422-436`
  - `tests/test_llm_client_empty_completion_retry.py:55-150`
  - `tests/test_c111_timeout_observability.py:362-396`
  - `docs/pwml_recovery_sprint/TEST_MATRIX.md:259-271`
- @ 82fc69a7:
  - `src/t2pw/mining/modelclient.py`
  - `src/t2pw/mining/config.py:174, 201`
  - `docs/pathway_mining/known_limitations.md:148-173`
  - `docs/pathway_mining/evidence/orch745/reviews/csc_r0.txt:42-68`
  - `docs/pathway_mining/evidence/013-item2-replay/call1_pilot_prompt_summary.json`
- `docs/pathway_mining/pilot/003/audit/PILOT-003-AUDIT.md` (experiment/pathway-mining-pilot-003)
- `docs/pathway_mining/pilot/002/audit/PILOT-002-AUDIT.md` (experiment/pathway-mining-pilot-002)
- `docs/pwml_recovery_sprint/ORCH-745-RESULT.md:166-173` (primary checkout)
- `C:/t/mining-handoff/MINING-OVERNIGHT-HANDOFF.md:100`
- DECISIONS.md D-102 (production frozen)
