# Evaluation v2 -- sample report (eval-v2.0 (ORCH-742c))

Run artifacts read: git-tracked files at commit `a0c0115acba51eb75f36c8c7749d9e72cc88b154` (the run trees are unchanged on this branch). Regenerate: `python -m t2pw.bench.eval_v2.report evaluate --repo . --out docs/evaluation_v2 --live-json docs/evaluation_v2/retrieval_live.json`.

> NEW evaluator version. Not a re-score, replacement or correction of T-107, T-108, T-109 or any committed benchmark result; those keep their meaning and were not re-run. Every number below is computed from git-tracked run artifacts at the commit this file was generated on. The mapped-ID reference tables (data/id_mapping_cache.json, data/enrichment_cache.json, data/pathwhiz_id_db.json) are mutable runtime caches; they are read as committed at HEAD, never from the working tree, and their blob ids are recorded under mapped_id_correctness.reference_tables. 'PWML produced' is not biological correctness and is not reported as such.

## 1. Newly complete positive gold cases

| case | in-scope reactions | off-pathway statements |
|---|---|---|
| PMC12096016 | 5 | 4 |
| PMC12312563 | 1 | 0 |
| PMC12421875 | 9 | 3 |
| PMC12657337 | 3 | 3 |

## 2. Reaction-level metrics (strict v2 matcher; micro-averaged; populations never summed)

| population | legs | precision | recall | F1 | unsupported rate (upper bound) | off-pathway rows | summary rows | name-conflict rows | partial rows | substrate acc | product acc | enzyme acc | species acc | evidence support | OLD lenient recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| canonical | 26 | 86.0% (117/136) | 96.6% (114/118) | 91.0% | 2.8% (4/141; <= 6.4%) | 5 | 1 | 0 | 9 | 96.5% | 97.9% | 100.0% | 100.0% | 77.3% | 90.7% |
| fallback | 44 | 64.2% (199/310) | 96.2% (177/184) | 77.0% | 16.8% (62/370; <= 33.2%) | 34 | 5 | 1 | 9 | 95.6% | 97.8% | 100.0% | 100.0% | 50.8% | 96.2% |
| T-107:canonical | 1 | 100.0% (5/5) | 100.0% (5/5) | 100.0% | 0.0% (0/5; <= 0.0%) | 0 | 0 | 0 | 0 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| T-107:fallback | 6 | 67.6% (25/37) | 92.3% (24/26) | 78.0% | 18.9% (7/37; <= 18.9%) | 4 | 0 | 0 | 1 | 90.6% | 96.2% | 100.0% | 100.0% | 56.8% | 92.3% |
| T-108:canonical | 2 | 100.0% (10/10) | 100.0% (10/10) | 100.0% | 0.0% (0/10; <= 0.0%) | 0 | 0 | 0 | 0 | 100.0% | 100.0% | 100.0% | 100.0% | 90.0% | 100.0% |
| T-108:fallback | 6 | 68.6% (24/35) | 92.3% (24/26) | 78.7% | 14.3% (5/35; <= 14.3%) | 5 | 1 | 0 | 0 | 90.6% | 92.3% | 100.0% | 100.0% | 68.6% | 92.3% |

Name-conflict rows for hand review (1): a parenthetical's inside names a gold term its outer name does not match, so the row is not credited (it counts toward the unsupported upper bound):

- `runs/2026-07-28_2122/papers/PMC12657337__high-level-production-of-vitamin-k2-in-lt-i-gt/research` row 17: ['heptanoyl pyrophosphate (HepPP)'] vs gold ['R1']

Old lenient matcher vs strict v2 matcher on the SAME rows (old credits / v2 refuses; v2 credits / old refuses): canonical: 0 / 7; fallback: 2 / 0; T-107:canonical: 0 / 0; T-107:fallback: 0 / 0; T-108:canonical: 0 / 0; T-108:fallback: 1 / 0

## 3. Mapped-ID correctness (every committed canonical leg)

canonical legs: 115. Each identifier is judged against its own record; 'resting on the mapper cache only' means no independent record (PathWhiz table, KEGG/ChEBI/HMDB/UniProt record) decided it. Reference tables are read as committed at HEAD (git blob ids: `data/enrichment_cache.json` a84eb220, `data/id_mapping_cache.json` 02b18edd, `data/pathwhiz_id_db.json` 75d2e5f4), never from the working tree, where the two caches are rewritten by every pipeline run.

| verdict | entities | of which resting on the mapper cache only |
|---|---|---|
| correct_identity | 1184 | 322 |
| not_evaluable_no_reference | 17 | 0 |
| unresolved_or_novel | 510 | 0 |
| wrong_identity | 119 | 47 |

## 4. RAG retrieval (live production retriever)

embedding_mode: **api** (text-embedding-nomic-embed-text-v1.5); corpus 10 committed paper texts, 232 chunks; top_k 8

| relevance label | positive queries | Recall@1 | Recall@3 | Recall@5 | Precision@5 | MRR |
|---|---|---|---|---|---|---|
| target_quote | 18 | 16.7% | 22.2% | 44.4% | 8.9% | 0.2926 |
| participants_named (any paper) | 18 | 27.8% | 55.6% | 61.1% | 18.9% | 0.437 |

Negative-query rejection: 0.0% (0/49). build_hybrid_scorer min-max normalises both halves WITHIN each query's pool, so the top hit scores ~1.0 whatever its relevance; the returned score cannot support a rejection floor, and the production retriever applies none.

Per retrieved chunk: provenance_present=True: 144, valid_source_reference=True: 144, validate_provenance_ok=True: 144

## 5. Retrieval vs LLM-use vs admission failure (archived admission reports)

| population | legs | already in Stage 1 | A not retrieved | B retrieved, not used | C proposed, rejected | admitted | accepted candidates | D accepted unsupported | accepted external biology |
|---|---|---|---|---|---|---|---|---|---|
| untruncated | 34 | 137 | 5 | 2 | 8 | 2 | 14 | 0 | 2 |
| truncated | 28 | 114 | 0 | 2 | 4 | 0 | 67 | 0 | 23 |

Per persisted candidate chunk: untruncated: provenance_present=True 3066, valid_source_reference=True 3066, validate_provenance_ok=True 3066; truncated: provenance_present=True 5667, valid_source_reference=True 5667, validate_provenance_ok=True 5667

## 6. Un-gated RAG influence audit (report only)

| artifact | legs | rows | rows absent from Stage 1 | un-gated RAG-influenced candidates | of which external retrieved evidence | legs with any |
|---|---|---|---|---|---|---|
| stage2_merge_output | 162 | 682 | 123 | 20 | 3 | 13 |
| canonical_final | 112 | 385 | 89 | 8 | 3 | 7 |

## 7. LLM reproducibility-metadata completeness

model_attempt records: 663 from 20 leg traces (runs: runs_verify/2026-09-01_1612); legs with a payload: 187. Audit rounds carrying temperature: 136/136.

| field | status | attempts with field | recorded as | future hook |
|---|---|---|---|---|
| requested_model | recorded | 663/663 | LEG_TRACE model_attempt.model (the model the client RESOLVED and sent) | src/t2pw/llm/client.py::chat_detailed -- already passed to record_model_attempt |
| served_model | not_recorded | 0/663 | not recorded (resp.model is never read) | src/t2pw/llm/client.py::chat_detailed, after _client.chat.completions.create: resp.model |
| temperature | not_recorded | 0/663 | audit_iteration_summary.json rounds[].temperature (audit stage only) | src/t2pw/llm/client.py::chat_detailed kwargs['temperature'] -> src/t2pw/batch/leg_trace.py::record_model_attempt (no parameter today) |
| seed | not_recorded | 0/663 | not recorded; no seed is sent (kwargs carry none) | src/t2pw/llm/client.py::chat_detailed kwargs (no evaluation-only config surface exists) |
| provider_backend_identity | not_recorded | 0/663 | not recorded (OpenRouter's response 'provider' is never read) | src/t2pw/llm/client.py::chat_detailed: getattr(resp, 'provider', None) |
| finish_reason | recorded | 663/663 | LEG_TRACE model_attempt.finish_reason | - |
| request_hash | recorded | 663/663 | LEG_TRACE model_attempt.request_hash = client._hash(messages): a hash of the MESSAGES only; model/temperature/max_tokens are not in it | src/t2pw/llm/client.py::chat_detailed (CompletionDiagnostics(request_hash=_hash(messages))) |
| prompt_hash | recorded | 663/663 | same value as request_hash (the only hash taken is over the messages) | - |
| response_hash | not_recorded | 0/663 | LEG_TRACE model_attempt.response_hash -- the key exists but is written EMPTY: CompletionDiagnostics.note() publishes the attempt row before chat_detailed assigns diagnostics.response_hash (client.py, the `diagnostics.response_hash = _hash(text)` line), so the published value is always '' | src/t2pw/llm/client.py::_publish_attempt / CompletionDiagnostics.note ordering |
| token_usage | not_recorded | 0/663 | not recorded per call; client._record_usage only accumulates process-wide totals in memory (_token_stats) | src/t2pw/llm/client.py::_record_usage -> record_model_attempt |
| retry_count | recorded | 663/663 | derived: max model_attempt.attempt per (stage, request_hash) | - |
