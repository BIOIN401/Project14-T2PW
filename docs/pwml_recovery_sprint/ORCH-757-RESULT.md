# ORCH-757 — one configurable, fallback-aware LLM token budget (D-105 § 2)

**Status:** implemented and gated on the lane branch; **not reviewed, not merged.** An independent
reviewer judges the actual diff next.

| | |
|---|---|
| branch | `orch757/llm-budgets` (worktree `.claude/worktrees/orch757`) |
| base | `sprint/pwml-recovery` @ `a43a2b12` |
| test-first commit (base code) | `089d3c7d` |
| code commit | `2aebbfe0` |
| docs/evidence commit | the commit that adds this file |
| charter | `DECISIONS.md` D-105 § 2; D-104 AMENDMENT 2 (LLM inputs byte-identical) |
| why | `ORCH-755-RESULT.md` § 4: `deepseek/deepseek-v4-flash` spent the small budgets reasoning and returned empty `finish_reason=length` completions |

No LLM call and no pipeline run was made. Every number below names its G11 report under
`evidence/g11/ORCH-757/` and its log under `evidence/orch757/`.

---

## 1. The resolver, the table, the clamp

All in `src/t2pw/llm/client.py` (every resolver user already imports this module, and the protected
`streamlit_app.py` already imports it as `llm_client_module`, so it needed no new import):

- **`configured_max_tokens(stage_env_var=None)`**: `stage_env_var`, else `LLM_MAX_TOKENS`, else
  **128,000** (`DEFAULT_MAX_TOKENS`). A non-integer or non-positive value logs a warning and gives
  128,000. This is `preprocessor._resolve_preprocess_max_tokens`'s logic, generalised; the
  preprocessor now delegates to it (same precedence; only its 10,000 default became 128,000).
- **`MODEL_MAX_COMPLETION_TOKENS`**, a config constant, never a network call. Measured 2026-10-21
  from OpenRouter `/api/v1/models` (values supplied with the dispatch):
  `google/gemini-3.8-flash` = 65,536; `deepseek/deepseek-v4-flash` = 943,717.
- **`UNLISTED_MODEL_MAX_COMPLETION_TOKENS`** = the smallest listed cap (65,536). See § 9 Q1.
- **`FAILOVER_MODEL_ENV_VARS`** = `{OPENROUTER_EXTRACTION_MODEL: OPENROUTER_EXTRACTION_FALLBACK_MODEL}`.
  The only failover in the code is the Stage-1 ladder's rung 3 (`extraction_ladder.ALTERNATE_MODEL_ENV_VAR`),
  which re-asks with the same budget. I checked for others: no OpenRouter `models` list,
  `extra_body`, or second transport exists (`grep -rn "completions.create\|extra_body\|\"models\"" src/`
  finds only the client and the stoich agent; `rag/embed.py` is embeddings).
- **`request_max_tokens(max_tokens, *, model, model_env_var=None)`**: `None` means configured; the
  result is `min(requested, cap(model), cap(fallback model))`, where the fallback model counts only
  when its variable is set (the same reachability rule as `alternate_model_env_var`) and is resolved
  with the client's own `_resolve_model`. A clamp is logged at INFO.
- `chat`, `chat_detailed`, `chat_with_tools` default `max_tokens=None`, and every request they build
  goes through `request_max_tokens`. The stoich agent calls the SDK directly, so it calls
  `request_max_tokens(None, model=_model)` itself.

Per-stage variables are unchanged: `OPENROUTER_PREPROCESSOR_MAX_TOKENS` (preprocessor default),
`OPENROUTER_EXTRACTION_MAX_TOKENS` and `OPENROUTER_INFERENCE_MAX_TOKENS` (the UI reads them through
`_bounded_env_int`, untouched; the pipeline's own defaults now read them too when a caller passes
nothing). Every other stage has no variable of its own and reads `LLM_MAX_TOKENS`.

With the operator's `.env` (all stages `deepseek/deepseek-v4-flash`, fallback `google/gemini-3.8-flash`,
`LLM_MAX_TOKENS=128000`, extraction 64,000): every stage requests 128,000; extraction requests
64,000 (under the 65,536 clamp); the Stage-1 row repair, which has no variable of its own, requests
65,536 (clamped, it runs on the extraction model).

## 2. Every call site, before and after

Line numbers are base `a43a2b12`.

| # | site | before | after |
|---|---|---|---|
| 1 | `streamlit_app.py:3816` audit round budget | `min(8000, 3600 + 700*(round-1))` | `llm_client_module.configured_max_tokens()` |
| 2 | `streamlit_app.py:3849` audit candidate | `min(10000, base + token_offsets[cand])` | `base_max_tokens` |
| 3 | `streamlit_app.py:3881` post-apply scoring audit (`use_llm=False`) | 1200 | `base_max_tokens` |
| 4 | `streamlit_app.py:3934` fallback deterministic audit (`use_llm=False`) | 1200 | `base_max_tokens` |
| 5 | `streamlit_app.py:3951` fallback scoring audit (`use_llm=False`) | 1200 | `base_max_tokens` |
| 6 | `streamlit_app.py:3973` fallback candidate record | `"max_tokens": 1200` | `"max_tokens": base_max_tokens` |
| 7 | `streamlit_app.py:4005` gap resolver | `min(1400, 700 + 120*(round-1))` | `llm_client_module.configured_max_tokens()` |
| 8 | `streamlit_app.py:4240` curator | 2000 | `llm_client_module.configured_max_tokens()` |
| 9 | `streamlit_app.py:4400` overwatch | 3000 | `llm_client_module.configured_max_tokens()` |
| — | `streamlit_app.py:3904`, `:4093` records | record `cand_max_tokens` / the selected candidate | unchanged lines; they now record the configured budget |
| — | `streamlit_app.py:5243`, `:5255` UI Stage 1/2 inputs | `_bounded_env_int(…, 64000, max 128000)` | **unchanged** (per-stage variables keep working as now; see § 9 Q4) |
| 10 | `llm/client.py:574` `chat` | 800 | `None` → configured, clamped |
| 11 | `llm/client.py:606` `chat_detailed` | 800 | `None` → configured, clamped |
| 12 | `llm/client.py:842` `chat_with_tools` | 1200 | `None` → configured, clamped |
| 13 | `pipeline.py:115` legacy SBML audit | 3600 | argument removed (callee default) |
| 14 | `pipeline.py:148` legacy SBML overwatch | 3000 | argument removed (callee default) |
| 15 | `pipeline.py:256` `run_extraction_pipeline` | 12000 | `None` → `OPENROUTER_EXTRACTION_MAX_TOKENS` → `LLM_MAX_TOKENS` → 128,000 |
| 16 | `pipeline.py:2656` `run_stage_one_with_chunking` | 12000 | `None` (passed through to #15) |
| 17 | `pipeline.py:397` `run_inference_pipeline` | 10000 | `None` → `OPENROUTER_INFERENCE_MAX_TOKENS` → … |
| 18 | `pipeline.py:440` `run_stage_two_with_chunking` | 10000 | `None` → same as #17 (resolved here: the retry arithmetic needs it) |
| 19 | `pipeline.py:686` `run_stage_two_with_feedback_loop` | 10000 | `None` (passed through) |
| 20 | `pipeline.py:589` `_default_retry_tokens` truncation branch | `min(24000, max(b+800, 1.5b))` | `max(b+800, 1.5b)`; the client clamps |
| — | `pipeline.py:3755`, `:4009` (`_run_json_stage` draw and JSON repair) | the stage's budget | unchanged |
| 21 | `preprocessor.py:68` default | 10,000 | 128,000 via `configured_max_tokens` |
| 22 | `rag/extract.py:48` `_MAX_TOKENS` | 1500 | constant removed; client default |
| 23 | `audit_json_llm.py:1406` `audit_payload` | 3200 | `None`, resolved in-function (its report records `llm.max_tokens`) |
| 24 | `audit_json_llm.py:1521` `run_audit` | 3200 | `None` |
| 25 | `audit_json_llm.py:1709` CLI `--max-tokens` | 3200 | `None` |
| 26 | `completeness_audit.py:314` | 4000 | `None` |
| 27 | `gap_resolver.py:298` `infer_entity_species` | 450 | `None` |
| 28 | `gap_resolver.py:2803` `resolve_gaps`, `:3388` `run_gap_resolution`, `:3449` CLI | 900, 900, 900 | `None` |
| 29 | `gap_resolver.py:3058` id-candidate choice | `max(500, int(b))` (floor) | `b` |
| 30 | `gap_resolver.py:3301` enrichment agent | `max(b, 1200)` (floor) | `b` |
| 31 | `gap_resolver.py:376, 845, 1524, 1648` | `int(max_tokens)` | `max_tokens` (the client coerces) |
| 32 | `interactive_curator.py:297` (round and its retry) | 4096 | `None` |
| 33 | `pathway_curator.py:214` | 2000 | `None` |
| 34 | `extraction/extract.py:18` demo | 1200 | argument removed |
| 35 | `map_ids.py:443` AI protein aliases | 300 | argument removed |
| 36 | `map_ids.py:3314` species via gap resolver | 450 | argument removed (callee default) |
| 37 | `focused_repair.py` `_call_repair_llm` and six pass functions | 4000 ×7 | `None` ×7 (no production caller today) |
| 38 | `localized_repair.py:1052` `repair_json_text` | 12000 | `None` (the pipeline passes the stage budget) |
| 39 | `localized_repair.py:1334` `repair_invalid_rows` | 4000 | `None` (`stage_one_boundary` passes none → configured, clamped by the extraction chain) |
| 40 | `sbml/overwatch.py:173`, CLI `:267` | 2600, 2600 | `None` |
| 41 | `stoich/agent.py:477` tool loop (direct SDK call) | **no `max_tokens` at all** (provider default) | `request_max_tokens(None, model=_model)` |
| 42 | `stoich/agent.py:568, 596` compound audits | 300, 300 | same as #41 |
| 43 | `stoich/classifier.py:136` | 300 | argument removed |
| 44 | `tools/pathwhiz_converter/llm_plan.py:58` | 2400 (`int()`) | `None` |

Found beyond the dispatch list: #41 (no budget at all), the three CLI defaults (#25, #28, #40), the
`int()` coercions (#31), the client's `chat_detailed` default (#11), the preprocessor default (#21).

**Sites 3–6 never reach a model** (`use_llm=False`). Changed because D-105 § 2 names them; the only
observable effect is the audit report's `llm.max_tokens` and the round record at `:3973`.

## 3. Retry and escalation ladders

| ladder | before | after | floor / ceiling |
|---|---|---|---|
| Stage-2 retry, truncation branch (`_default_retry_tokens`) | `min(24000, max(b+800, 1.5b))`: with the UI's 64,000 it **shrank** to 24,000 | `max(b+800, 1.5b)` | no absolute ceiling; the model cap via the client (deepseek 943,717) |
| Stage-2 retry, other failures | `max(200, 0.6b)` | unchanged (relative shrink ladder, allowed to stay) | floor 200 |
| audit candidates within a round | `min(10000, base + [0,500,900,1300,1700])` | every candidate = configured budget; candidates still differ by temperature | none |
| audit rounds | base `min(8000, 3600+700(r-1))` | configured budget every round | none |
| gap resolver rounds | `min(1400, 700+120(r-1))` | configured budget every round | none |
| Stage-1 ladder (draws, repair, rung 3) | one budget for every rung | unchanged; rung 1 and repair now clamped to the fallback's cap too | model caps |
| chat retry loops (`LLM_MAX_RETRIES`) | same request re-sent | unchanged | — |

The first request of every stage equals its configured, clamped budget. `token_offsets` at
`streamlit_app.py:3818` is now unused; it stays because no other byte of that file may change.

## 4. `streamlit_app.py`: the exact change

Applied by `evidence/orch757_streamlit_budget_edit.py`: byte-level replacements on the CRLF file,
each old line matched an exact number of times, then asserted that exactly 9 lines differ, the line
count is equal and no bare LF appeared. `git diff a43a2b12 -- src/t2pw/app/streamlit_app.py`:

```diff
@@ -3813,7 +3813,7 @@ def run_post_pipeline_sbml_artifacts(
             round_input_hash = _payload_file_hash(current_input)
 
             base_temperature = min(0.65, 0.15 * (round_idx - 1))
-            base_max_tokens = min(8000, 3600 + 700 * (round_idx - 1))
+            base_max_tokens = llm_client_module.configured_max_tokens()
             temp_offsets = [0.0, 0.14, 0.28, 0.40, 0.50]
             token_offsets = [0, 500, 900, 1300, 1700]
             candidate_count = 1 if not use_llm_audit else min(max_candidates, len(temp_offsets))
@@ -3846,7 +3846,7 @@ def run_post_pipeline_sbml_artifacts(
                     timed_out_mid_round = True
                     break
                 cand_temperature = min(0.9, base_temperature + temp_offsets[cand_idx])
-                cand_max_tokens = min(10000, base_max_tokens + token_offsets[cand_idx])
+                cand_max_tokens = base_max_tokens
                 cand_suffix = f"round{round_idx}.cand{cand_idx + 1}"
                 cand_audit_report = tmp / f"audit_report.{cand_suffix}.json"
                 cand_audit_patch = tmp / f"audit_patch.{cand_suffix}.json"
@@ -3878,7 +3878,7 @@ def run_post_pipeline_sbml_artifacts(
                     cand_eval_patch,
                     use_llm=False,
                     llm_temperature=0.0,
-                    llm_max_tokens=1200,
+                    llm_max_tokens=base_max_tokens,
                     context_note="deterministic post-apply scoring",
                 )
 
@@ -3931,7 +3931,7 @@ def run_post_pipeline_sbml_artifacts(
                     cand_audit_patch,
                     use_llm=False,
                     llm_temperature=0.0,
-                    llm_max_tokens=1200,
+                    llm_max_tokens=base_max_tokens,
                     context_note="fallback deterministic audit after timeout/empty candidate set",
                     retrieval_context="",
                 )
@@ -3948,7 +3948,7 @@ def run_post_pipeline_sbml_artifacts(
                     cand_eval_patch,
                     use_llm=False,
                     llm_temperature=0.0,
-                    llm_max_tokens=1200,
+                    llm_max_tokens=base_max_tokens,
                     context_note="deterministic fallback scoring",
                     retrieval_context="",
                 )
@@ -3970,7 +3970,7 @@ def run_post_pipeline_sbml_artifacts(
                     {
                         "index": 1,
                         "temperature": 0.0,
-                        "max_tokens": 1200,
+                        "max_tokens": base_max_tokens,
                         "score": list(score),
                         "audit_error_count": int(cand_audit_summary.get("error_count", 0)),
                         "audit_warning_count": int(cand_audit_summary.get("warning_count", 0)),
@@ -4002,7 +4002,7 @@ def run_post_pipeline_sbml_artifacts(
 
             if use_gap_resolver and not timed_out_mid_round:
                 gap_temp = min(0.45, 0.05 + 0.08 * (round_idx - 1))
-                gap_tokens = min(1400, 700 + 120 * (round_idx - 1))
+                gap_tokens = llm_client_module.configured_max_tokens()
                 gap_report_round = run_gap_resolution(
                     round_audited,
                     round_resolved,
@@ -4237,7 +4237,7 @@ def run_post_pipeline_sbml_artifacts(
                 pathway_name=str(_meta.get("pathway_name", "") or ""),
                 organism=str(_meta.get("organism", "") or ""),
                 llm_temperature=0.2,
-                llm_max_tokens=2000,
+                llm_max_tokens=llm_client_module.configured_max_tokens(),
             )
             # Use curated output as input to mapping only if patches were accepted
             if int(_safe_dict(curator_report.get("summary", {})).get("patches_accepted", 0)) > 0:
@@ -4397,7 +4397,7 @@ def run_post_pipeline_sbml_artifacts(
                     sbml_report_json_path,
                     sbml_overwatch_path,
                     use_llm=True,
-                    llm_max_tokens=3000,
+                    llm_max_tokens=llm_client_module.configured_max_tokens(),
                 )
             try:
                 render_artifacts = build_render_artifacts(str(sbml_path))
```

Nine lines, each a token-budget expression at an LLM call site or the budget record beside one. No
import, no other byte. `llm_client_module` is the file's existing `import t2pw.llm.client as
llm_client_module` (line 25).

**Fingerprints** (sha256 unless stated):

| | working tree (CRLF, what the app runs) | git blob content (LF) | git blob id |
|---|---|---|---|
| before (`a43a2b12`) | `7bf2aea33272160665bfee136df57d60a4a30b82b5d54a916394ea4b0ff05c2f` | `d4aa606a534dffaab654f6ecaa8598859856754a45c168e9530572711e5ccdf1` | `66119e25a3cec7de60a5ded8558877f31db5c2e4` |
| after (`2aebbfe0`) | `28c091171f18a700dea952723f5451f012090640712c0821ac66d58134f892ae` | `52e016d7e823122a1762dd0e26c475950ce58331598c79aa5e15385aa765dd7c` | `12bf51735565a02f632734cdff475a97d44240a0` |

The "before" values equal `ORCH-754a-RESULT.md` § 11's "after" values. No test hashes this file.

## 5. Proofs

**G9 (behavioural, base-failing):** `tests/test_orch757_llm_token_budgets.py`, committed first on the
base code (`089d3c7d`). Each of 23 LLM call sites is driven through its real entry point with
`t2pw.llm.client._client` faked, so every asserted number is the `max_tokens` on the wire:

- `LLM_MAX_TOKENS=50000` → every request of every stage asks for 50,000 (23 cases);
- nothing configured → 128,000 (23 cases); an unusable value (`lots`, `0`, `-5`) → 128,000;
- `OPENROUTER_EXTRACTION_MAX_TOKENS` / `OPENROUTER_INFERENCE_MAX_TOKENS` feed the pipeline defaults;
- extraction on deepseek with the gemini fallback set: 128,000 configured, or passed explicitly the
  way the UI does → **65,536**; the row and JSON repairs on the extraction model likewise; rung 3's
  own call on gemini → 65,536;
- an unlisted model → 65,536 (and a smaller configured budget passes untouched);
- the Stage-2 truncation retry, driven with ORCH-755's measured shape (an empty completion with
  `finish_reason=length`): base `[64000, 64000, 24000, 24000]`, tip 64,000 then 96,000;
- the protected-file sites, through the real `run_post_pipeline_sbml_artifacts` source executed by
  `tests/test_streamlit_stage2_orchestration.py`'s harness: both LLM audit candidates, both scoring
  audits, the gap resolver, curator and overwatch receive the configured budget, and
  `audit_iterations[0]["max_tokens"]` records it. A fake clock reaches the timed-out-round fallback,
  so sites 4–6 are measured too. **All 9 changed lines are reached.** There the number asserted is
  the argument the call site hands its stage; each stage's own wire test is in the list above.

| run | commit / tree | report | result |
|---|---|---|---|
| RED | `089d3c7d` (base code) | `g11/ORCH-757/05` | **52 failed, 3 passed** |
| GREEN | `2aebbfe0` | `g11/ORCH-757/07` | **55 passed** |

The 3 that pass at base are not G9 cases: the two preservation tests, and the parametrised
`preprocessor` case under `LLM_MAX_TOKENS`, which already worked at base (the preprocessor read that
variable). The preprocessor's G9 cases are the nothing-configured and unusable-value ones.

**No change to LLM-bound content (D-104 AMENDMENT 2), preservation, not G9:**
`test_requests_are_byte_identical_except_the_budget` pins the sha256 of every request each of the 23
stages sends, minus `max_tokens` and `timeout` (so messages, model, temperature, response format,
JSON schema and tools are all covered), to values measured at base by
`evidence/orch757_golden_capture.py`: two base captures in different scratch directories
(`/03`, `/04`) were identical, and the tip capture (`/26`) is byte-identical to them (both files
sha256 `f8f6cdd8dc8523577134156dbd87cfbf92189a761cfce9fbe062011d9db51ddb`). It passes at base (`/05`)
and at the tip (`/07`). Not covered by the golden: rung 3 through the ladder (needs a live leg
deadline; its prompt code is untouched).

The test pins every variable it relies on (`LLM_MAX_TOKENS`, clears the three per-stage budgets and
eleven model selectors, `T2PW_OFFLINE_CURATOR`) and forces `PROVIDER=openrouter`, so the primary
checkout's `.env` cannot change its result.

## 6. Pins moved (merge rule 4)

**None.** Every existing test that names a budget passes it explicitly (`test_completeness_audit` 5000,
`test_curator_offline_mode` 2000, `test_stage_one_ensemble` 12000, the ladder/repair tests 100,
`test_focused_repair` 4000, `test_preprocessor` 10000/12000/64 via env or argument), and explicit
values are honoured unchanged. One harness changed: `tests/test_streamlit_stage2_orchestration.py`
`_orchestration_harness` now injects the real `t2pw.llm.client` as `llm_client_module` (inert at base;
needed at the tip). `test_c030a…` and `test_c052…` reuse that harness.

## 7. Gates

| gate | report | result |
|---|---|---|
| 1. focused: G9 RED at base code | `/05` | 52 failed, 3 passed |
| 1. focused: G9 + golden GREEN at tip | `/07` | 55 passed |
| 1. golden captures base ×2, tip | `/03`, `/04`, `/26` | identical |
| 2. affected chunk 0 (22 files) | `/08` | 416 passed, 1 skipped, 1 failed: `test_c075…::test_corpus_…` (pre-existing) |
| 2. affected chunk 1 (26 files) | `/09` | 614 passed, 5 skipped, 3 failed: `test_c100…::test_5b_…`, `test_compound_resolution_extraction.py` ×2 (pre-existing) |
| 2. affected chunk 2 (26 files) | `/10` | 4 collection errors: order hazard, identical on base `src/` (`/11`); see F2 |
| 2. affected chunk 2 reordered | `/12` | 544 passed, 3 failed (`test_llm_client_request_timeout.py`, order-dependent; same 3 on base `src/`, `/13`); the file alone: 6 passed (`/14`) |
| 2. affected chunk 3 (26 files) | `/15` | 641 passed, 1 failed: `test_rag_seed_entity_reimport…::test_the_orchestrator_passes_the_merge_base_to_conform` (pre-existing) |
| 2. the 5 pre-existing failures on base `src/` | `/16` | 5 failed |
| 3. SMOKE | `/17` | **508 passed** |
| 4. `tests/test_eval_v2_*.py` | `/18` | 89 passed, 1 xfailed |
| 5. `test_batch_preflight.py` alone | `/19` | 32 passed, 4 skipped, 1 failed: `test_the_message_names_the_problem_the_interpreter_and_the_cure` ("this project ships a .venv": a worktree has none); same on base `src/` (`/22`) |
| 5. `test_c055_rag_loop_wiring.py` alone | `/20` | 14 passed |
| 5. `test_streamlit_quarantine_boundary.py` alone | `/21` | 23 passed |
| 5. `test_c052_prefreeze_report_at_the_streamlit_seams.py` alone | `/23` | 13 passed |
| 5. `test_reaction_evidence_view.py` alone | `/24` | 11 passed |
| 5. `test_reaction_evidence_app_hook.py` alone | `/25` | 1 passed |
| exploration on base (before the test-first commit) | `/01`, `/02` | 53 failed / 2 passed; 4 failed |
| G9 on the uncommitted tip (same bytes as `2aebbfe0`) | `/06` | 55 passed |

Every report: `FINAL SURVIVING COUNT : 0`. `g11_evidence.py check --task ORCH-757` (exit 0): 26
artifacts, 0 non-compliant; 23 invoke pytest directly, all 23 with a committed pin verdict, none
refused, no foreign `src`; the other 3 (`/03`, `/04`, `/26`) are the golden-capture script.

The affected set is every test file that imports a changed module or names `max_tokens`, plus every
file that drives `streamlit_app.py`, minus SMOKE (gate 3), `test_eval_v2_*` (gate 4) and the six
run-alone AppTest files (gate 5): 100 files, listed in `evidence/orch757/affected_chunk_0*.txt`.

"On base `src/`" means `git checkout a43a2b12 -- src/` in this worktree with the tip's tests, then
`git checkout HEAD -- src/` (no stash). The logs carry base line numbers (`client.py:746`,
`pipeline.py:511`) where the tip's carry `:809` / `:515`. Skip counts include the path-sensitive
c081/c089 corpus skips expected under a `worktrees` path.

## 8. Findings, failures and dead ends

- **F1. The Stage-2 truncation escalation is unreachable after a failed JSON repair.** Pre-existing;
  measured in `/01`: a non-JSON reply gave retries at 38,400 (0.6 × 64,000), not the truncation
  branch. `_summarize_failure` reads the last attempt with an error, which is the `json_repair` row
  (`"json repair exhausted its N-attempt cap: …"`), so `_looks_truncated_json_failure` never matches.
  Only an empty reply (no repair attempted) reaches the 1.5× branch, which is ORCH-755's actual
  failure shape and what the G9 test drives. Not changed: the detector is not a budget. § 9 Q3.
- **F2. `tests/test_gap_resolver*.py` (four files) install a stub `t2pw.llm.client` with
  `sys.modules.setdefault` at import.** Collected before anything imports the real client, they break
  every later file that imports `chat_detailed`, `CompletionDiagnostics` or `_completion_is_empty`.
  Measured identical on base `src/` (`/10` vs `/11`). The stubs define only `chat` and
  `chat_with_tools`, and every module that now imports a new client name already imported one the stub
  lacks (`PROVIDER`, `chat_detailed`, `_client`), so this lane adds no new failure surface.
  `gap_resolver.py` deliberately imports nothing new. Separately, `test_llm_client_request_timeout.py`
  fails 3 cases when batched after those files and passes alone (`/12`, `/13`, `/14`).
- **F3. The stoich agent's tool-loop request carried no `max_tokens` at all** (the provider's own
  default). It now sends the configured budget.
- **F4. Dead end, retry proof.** My first G9 for the retry used a non-JSON reply and measured the
  0.6× shrink branch at base, which is how F1 was found. The test now uses the empty `length` reply.
- **F5. Dead end, harness.** The first app-site run asserted two audit candidates and got one: with
  fewer than 120 s left the app drops to one candidate, and the harness default timeout is 120 s. The
  test passes `audit_timeout_seconds=600`.
- **F6. Dead end, line endings.** `sed -i` from Git Bash rewrote `preprocessor.py` from CRLF to LF. I
  restored it byte-for-byte before committing; every other edit, the protected file included, was made
  by counted byte-level replacement scripts that keep CRLF.

## 9. Not settled: for the product owner

- **Q1. A model not in the table.** Implemented: it gets the smallest listed cap, 65,536. This
  includes the local LM Studio default `meta-llama-3.1-8b-instruct` and any new OpenRouter model.
  Options:
  - (a) keep it. An unlisted model with a larger cap is held to 65,536 until a code change adds a row.
    One with a cap below 65,536 can still get a provider 400 on that stage.
  - (b) pass the configured budget through unclamped. That gives full budgets, but any unlisted
    failover can 400.
  - (c) a table read from configuration. That adds one more knob, and a typo there is silent.
- **Q2. A truncation retry may exceed the configured budget.** It asks for 1.5× (inference on deepseek:
  128,000 → 192,000), bounded only by the model cap. The alternative is to cap retries at the
  configured budget. Then a truncation retry asks for the same budget that just truncated.
- **Q3. F1.** Making the truncation detector see through a failed repair would make the 1.5× branch
  reachable for non-empty truncated replies. That changes retry behaviour beyond D-105 and needs its
  own decision.
- **Q4. The UI's own defaults stay at 64,000** (`streamlit_app.py:5243`, `:5255`, both bounded at
  128,000) when `OPENROUTER_EXTRACTION/INFERENCE_MAX_TOKENS` are unset. With nothing configured the
  UI's Stage 2 therefore asks for 64,000, not 128,000. Changing that is one literal in the protected
  file. I left it because the dispatch said those variables keep working exactly as now.
- **Q5.** With `.env`'s `LLM_MAX_TOKENS=128000`, the Stage-1 row repair (no variable of its own, on the
  extraction model) asks for 65,536, while extraction itself asks for 64,000. Should the row repair
  follow `OPENROUTER_EXTRACTION_MAX_TOKENS`? That adds one stage-variable mapping.
- **Q6. The cap table is a 2026-10-21 snapshot.** If OpenRouter lowers a cap, requests 400 until the
  table is edited. No runtime check exists, by design (dispatch: config constant, never a network
  call).
- Not measured: any live call. The 3-draw re-run (ORCH-759) is the acceptance test.

## 10. TEST_MATRIX delta (for the orchestrator to apply; I did not edit it)

Append end-of-file:

```markdown
## ORCH-757 (D-105 § 2): test obligations — appended end-of-file by rule · 2026-10-21

Lane `orch757/llm-budgets` (`089d3c7d` test-first, `2aebbfe0` code). Reports `evidence/g11/ORCH-757/01`–`26`.

**SMOKE is unchanged at 508** (`g11/ORCH-757/17`). No pin moved.

### New focused selection

`tests/test_orch757_llm_token_budgets.py` (55): the `max_tokens` on the wire for 23 LLM call sites and the
protected `streamlit_app.py` budget sites, plus a golden of every request minus `max_tokens`/`timeout`
pinned at base `a43a2b12`. Any prompt or message change to a covered stage fails the golden until it
is re-pinned with `evidence/orch757_golden_capture.py`. Run it for any change to `src/t2pw/llm/client.py`
or to an LLM call site. It pins its own budget and model variables; `.env` cannot change it.

### Harness

`tests/test_streamlit_stage2_orchestration.py::_orchestration_harness` injects the real `t2pw.llm.client`
as `llm_client_module`. Any harness that execs `run_post_pipeline_sbml_artifacts` into its own namespace
must provide it.

### Collection-order hazard (pre-existing)

`tests/test_gap_resolver.py`, `_agent_tools.py`, `_lineage.py` and `_stage3_issues.py` install a stub
`t2pw.llm.client` with `sys.modules.setdefault` at import. Collected first in a batch, they cause
`ImportError` in later files that import the real client (`test_preprocessor.py`,
`test_pipeline_cleanup.py`, `test_pipeline_lineage_carrier.py`, `test_llm_client_empty_completion_retry.py`).
Batched after them, `test_llm_client_request_timeout.py` fails 3 cases. It is identical on base
(`g11/ORCH-757/10`/`11`, `12`/`13`). Collect the gap-resolver files last, or run them in their own job.

### Known failures

Unchanged from ORCH-754's list. Measured red on base `src/` again in `g11/ORCH-757/16`. In a worktree,
`test_batch_preflight.py::test_the_message_names_the_problem_the_interpreter_and_the_cure` also fails
(no `.venv` in a worktree; same on base, `/22`).
```

## 11. Files changed

Code (`2aebbfe0`): `src/t2pw/llm/client.py`, `src/t2pw/app/streamlit_app.py` (9 lines),
`src/t2pw/pipeline/{pipeline,preprocessor,focused_repair,localized_repair}.py`,
`src/t2pw/curation/{audit_json_llm,completeness_audit,gap_resolver,interactive_curator,pathway_curator}.py`,
`src/t2pw/sbml/overwatch.py`, `src/t2pw/extraction/extract.py`, `src/t2pw/mapping/map_ids.py`,
`src/t2pw/rag/extract.py`, `src/t2pw/stoich/{agent,classifier}.py`,
`src/t2pw/tools/pathwhiz_converter/llm_plan.py`.

Tests (`089d3c7d`): `tests/test_orch757_llm_token_budgets.py` (new),
`tests/test_streamlit_stage2_orchestration.py` (harness namespace), and
`docs/pwml_recovery_sprint/evidence/orch757_golden_capture.py`.

Docs and evidence (this commit): this file, `evidence/orch757_streamlit_budget_edit.py`,
`evidence/g11/ORCH-757/` (26 reports), `evidence/g11/pin/ORCH-757/`, and `evidence/orch757/`
(logs and file lists).

Not edited: `TEST_MATRIX.md`, `LEDGER.md`, `DECISIONS.md`.
