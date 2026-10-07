# Evaluation v2 — readiness for the upcoming results run (ORCH-742c)

## Verdict

**READY WITH GAPS:** (1) reaction precision/recall/F1 are measurable only on the three certified-complete gold
papers (PMC12657337 is labelled but not certified complete, D-103) — any other paper needs its own exhaustive
gold first; (2) per-call LLM reproducibility metadata is
incomplete at the source — 6 of 11 fields are not recorded (served model, per-call temperature, seed,
provider identity, token usage, and a response hash that is written EMPTY on every committed record), and
closing that needs production hooks this lane may not add; (3) the mapped-ID `wrong_identity` flag is a
candidate list with a measured precision of 5/21 on T-107/T-108 and needs adjudication or a synonym source
(47 of the 119 corpus flags rest on the mapper's own candidate cache only, and the PathWhiz table is an
authority only for its records that carry an external identifier);
(4) retrieval-level negative-query rejection is structurally 0% (the production retriever has no score floor
and normalizes scores per query), and the archived A/B split sees only persisted candidate spans;
(5) the fallback-embedding cache defect is confirmed and unfixed, so a results run made during an embedding
outage would silently poison later runs' retrieval; (6) the mapped-ID reference tables
(`data/id_mapping_cache.json`, `data/enrichment_cache.json`, `data/pathwhiz_id_db.json`) are mutable runtime
caches that every pipeline run rewrites and the sprint never commits: the evaluator pins them by reading them as
committed at HEAD and recording each blob id in the report (today 02b18edd / a84eb220 / 75d2e5f4), so a results
run's mappings are judged against that committed snapshot (plus each payload's own enrichment sub-records and
mapping candidates), not against the caches the run rewrote; an identifier the snapshot lacks is `no_reference`,
and refreshing the snapshot means committing a cache, which is the product owner's call.

## What is ready, with the measured sample (from `SAMPLE_REPORT.md`)

| capability | status | sample |
|---|---|---|
| exhaustive gold for positive cases | 3 cases certified complete, 15 in-scope reactions, every quote verified in the fetched text and every in-scope quote in the committed text; a 4th case is labelled but not certified (D-103) and is not scored | PMC12096016 (5), PMC12312563 (1), PMC12421875 (9); PMC12657337 (3) labelled only |
| reaction P / R / F1, two tables (D-093) | ready for those 3 papers | canonical, 26 legs: P 86.0% (117/136), R 96.6% (114/118), F1 91.0%; fallback, 30 legs: P 70.0% (145/207), R 95.1%, F1 80.7% |
| unsupported-reaction rate | ready; lineage-aware, upper bound reported | canonical 2.8% (4/141, ≤ 6.4%); fallback 21.5% (48/223, ≤ 28.7%; no `name_conflict` row remains, its case being unscored under D-103) |
| entity accuracy | ready | canonical substrate 96.5%, product 97.9%, enzyme 100% of matched, species 100% of known (12 unknown) |
| evidence support | ready | canonical 77.3% of rows carry a verbatim paper quote; fallback 51.6% (53 rows carry only document dumps) |
| strict vs old matcher | reported side by side on the same rows; a parenthetical's inside never overrides a contradicting outer name (the one committed `name_conflict` row belongs to PMC12657337, unscored since D-103, and stays pinned by a test) | canonical old recall 90.7% vs v2 96.6% (7 rows the old matcher refused on an enzyme-name contradiction); fallback 2 rows credited by old only through congener containment (`DHNA` for `DHNA-CoA`) |
| mapped-ID correctness | ready as a candidate flag, judged per identifier, against the reference tables as committed (blob ids in the report) | 115 canonical legs: 1184 correct (322 on the mapper cache only), 510 unresolved/novel, 119 wrong-identity candidates (47 on the mapper cache only), 17 not evaluable |
| live retrieval (production retriever, `embedding_mode: api`) | ready; measured before D-103 and not re-measured, so 3 of 18 positive and 24 of 49 negative queries involve PMC12657337 | target-quote Recall@1/3/5 16.7% / 22.2% / 44.4%, P@5 8.9%, MRR 0.293; participants-named Recall@5 61.1%, MRR 0.437; negative rejection 0/49 |
| retrieval vs LLM-use vs admission (A/B/C/D) | ready on archived reports | untruncated 32 legs: A 5, B 2, C 8, admitted 2, D 0 (131 already in Stage 1); truncated 18 legs: B 2, C 4 |
| chunk provenance | ready | 144/144 live hits and 6,407/6,407 persisted candidates: provenance present, `validate_provenance` ok, valid source reference |
| un-gated RAG influence audit | ready, report-only | Stage-2 merge output: 20 candidates in 13 of 162 legs (3 on external retrieved evidence); canonical final: 8 in 7 of 112 (3 external) |
| reproducibility-metadata completeness | measured | 663 committed model_attempt records (T-108 only): model, finish_reason, request(=prompt) hash, retry count 100%; the other 6 fields 0% |

## Gap detail, with the exact site of each future hook (none added here)

| gap | site |
|---|---|
| served model | `src/t2pw/llm/client.py::chat_detailed`, after `_client.chat.completions.create`: `resp.model` |
| per-call temperature | `chat_detailed` `kwargs['temperature']` → `src/t2pw/batch/leg_trace.py::record_model_attempt` (no parameter today) |
| seed | `chat_detailed` kwargs; no evaluation-only configuration surface exists, so none was forced |
| provider identity | `chat_detailed`: OpenRouter's `provider` field on the response |
| response hash | `client.py` `CompletionDiagnostics.note` → `_publish_attempt` runs before `diagnostics.response_hash` is assigned |
| token usage | `client.py::_record_usage` only accumulates process totals |
| fallback embedding cache | `src/t2pw/rag/embed.py::Embedder.embed_one` caches `_embed_uncached`'s lexical fallback under the model key |
| identity synonyms | ChEBI enrichment: all 265 cached lookups are errors (`data/enrichment_cache.json`) |

## What this verdict does not say

It does not accept, reject or re-score T-107, T-108 or T-109; it does not claim the pipeline is biologically
correct because a PWML was produced; and the gap list is not a request to change production — each item is the
product owner's call.
