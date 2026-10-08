# PF-746-04 — rag/embed: lexical-fallback vectors are cached under the real model identity

Status: REGISTERED, not dispatched. Production is frozen (D-102); dispatch needs a product-owner decision.

## Defect

`Embedder.embed_one` caches whatever `_embed_uncached` returns. The key is built from the configured model name and the text, and the cache does not record which path produced the vector (`src/t2pw/rag/embed.py:111-121` at `b4bb0b74`):

```python
key = _cache_key(self.model, text)          # :112
...
vec = self._embed_uncached(text)            # :118
self._cache[key] = vec                      # :119
```

- **Fallback path.** `_embed_uncached` returns the deterministic bag-of-tokens vector whenever the API call yields `None` (`embed.py:126-133`). `_embed_via_api` turns every exception into `None` (`embed.py:165-166`, `except Exception: ... return None`).
- **Poisoning.** When the embeddings endpoint (LM Studio) is down, the lexical vector is stored under `_cache_key(<real model>, text)` (`embed.py:41-43`). `save()` writes it to disk (`embed.py:168-176`), by default to `<checkout>/data/rag_index/embeddings_cache.json` (`embed.py:179-183`; `config.py:142`).
- **Later runs.** With the endpoint back up, that vector comes back as a cache hit (`embed.py:113-116`), and the text is never re-embedded.
- **Offline runs.** The same thing happens for `online=False` with a real model configured, because the key ignores `self.online` (`embed.py:112`).

Nothing downstream catches this when widths match:

- `MemoryVectorStore.query` switches to lexical overlap only when a vector is missing or the query and chunk widths differ (`src/t2pw/rag/store.py:222-235`).
- The default backend, chroma, has no such check. `ChromaVectorStore.query` passes the query vector straight to chroma (`store.py:353-361`).

## Observed impact

- **Shown in tests, not in a live run.**
  - `tests/test_eval_v2_embedding_cache.py:37-50` pins today's behaviour: the fallback vector is served as a hit under the model key, with stats `{"hits": 1, "misses": 0, "api": 0, "lexical": 0}`.
  - `:53-63` is a `strict=True` xfail that encodes the correct behaviour.
  - ORCH-742 records this as X-2, "confirmed" (`docs/pwml_recovery_sprint/ORCH-742-RESULT.md:126`).
  - `docs/evaluation_v2/README.md:259-266` adds that with `RAG_EMBEDDING_DIM` unset, the fallback is 256 wide against the model's 768, so `MemoryVectorStore.query` quietly scores those pairs by lexical overlap. With matching widths it cannot tell the difference at all.
  - These tests were not run for this card (read-only).
- **Mining pilots 001-003: no observed impact.**
  - All three record `Embedding mode: api` (`pilot/001/run/report.md:80`, `pilot/002/run/report.md:116`, `pilot/003/run/report.md:88` on their pilot branches). On its own, `api` cannot rule out poisoned hits mixed with live API misses (see the mitigation section).
  - Each pilot resolved `PROJECT_ROOT` to its own worktree: `C:\t\mining-pilot`, `-pilot2`, `-pilot3` (`pilot-00N.log:1`).
  - None of those worktrees has `data/rag_index/embeddings_cache.json` today, and mining never saves one. So the pilots most likely ran on an empty embedding cache. This is inferred: the directory state at run time is not recorded.
  - Pilot-003's preflight probe shows `{"api": 1, "hits": 0, "lexical": 0, "misses": 1}` (`pilot/003/run/preflight.json`, `embeddings.reachable`).
  - The PILOT-002 and PILOT-003 audits do not mention an embeddings outage.
- **Inferred, not observed: production poisoning.** A production research-mode run made while LM Studio is down, with a real model configured, would persist poisoned entries:
  - The path is `streamlit_app.py:578-584` → `ingest.py:586`, then `embedder.save()` at `ingest.py:623-624`.
  - Every later run in that checkout would then be served those entries.
- **Primary checkout cache, inspected read-only for this check.**
  - The file's `model` field is `text-embedding-nomic-embed-text-v1.5`, and it was last written 2026-09-17.
  - It holds 5,385 entries: 5,199 are 768 wide, 47 are 256 wide and 139 are 64 wide.
  - The non-768 entries have lexical-fallback widths. But the key is a SHA-256 of model and text, and the file-level `model` field only records the last saver (`embed.py:171`). So whether any of them sit under the model key, and are therefore poisoned, cannot be determined.
  - The directory is gitignored (`.gitignore:24`).

## Current mitigation in mining

Partial. Mining detects a live outage but not poisoned cache hits.

- **Preflight probe.** The preflight probes the endpoint, and `PATHWAY_MINING_REQUIRE_EMBEDDINGS` turns an unanswered probe into a FAIL (`src/t2pw/mining/preflight.py:153-195`, `:295-297`, `:307` at `82fc69a7`).
- **Run-time mode.** `PassageIndex._record_embedding_mode` sets `embedding_mode` from the embedder's `api` and `lexical` counters (`src/t2pw/mining/retrieval.py:322-341`).
  - It raises `EmbeddingsUnavailable` only when `require_embeddings` is set.
  - The controller builds `PassageIndex(self.config)` without that argument (`src/t2pw/mining/controller.py:222`), so the env var takes effect only through the preflight.
- **Blind spot (read from the code, not observed).** A poisoned hit increments only `hits` (`embed.py:113-116`), so neither counter moves. A run served only poisoned hits would record `unknown`, and a run that mixes poisoned hits with API misses would record `api`.
- **Cache use.** Mining reads the checkout's default cache: `Embedder()` at `retrieval.py:305-310` and `preflight.py:176`. No `.save(` call appears anywhere under `src/t2pw/mining`, so mining never writes that cache.
- **Documentation.** `docs/pathway_mining/known_limitations.md:42-49` documents the outage case only.

## Proposed change

The fix is in one function, `Embedder.embed_one` (`src/t2pw/rag/embed.py:111-121`). A cached vector should be keyed by what produced it:

1. When the embedder will try the API (`self.online and self.model`; see `embed.py:127`, `:158`), look up and store under `_cache_key(self.model, text)`, as today.
2. Otherwise (offline, or no model), use a distinct lexical identity, e.g. `_cache_key(f"lexical:{self.dim}", text)`. Offline runs keep their cache hits and can never write a model key.
3. If the API was attempted but the vector came from the fallback (`self.stats["api"]` unchanged across `_embed_uncached`), return it **without caching** and without marking the cache dirty.

`_embed_uncached`, `_embed_via_api`, `save` and `_load_cache` stay unchanged.

This prevents new poisoning only. Entries already poisoned in a checkout cannot be told apart at matching width, and they keep being served.

Operational remediation, not a code change: after the fix lands, delete each checkout's `data/rag_index/embeddings_cache.json` once. Also delete any persisted vector index built from it; the default backend is `chroma` (`src/t2pw/config.py:141`), which persists under the same index dir (`store.py:423-427`, `:446`).

## Ownership

- `src/t2pw/rag/embed.py :: Embedder.embed_one` (lines 111-121 at `b4bb0b74`). No other production lines.
- `tests/test_eval_v2_embedding_cache.py`:
  - Remove the xfail marker (`:53-57`).
  - Rewrite or remove the characterisation test that asserts the defect (`:37-50`). After the fix, its assertion at `:45` fails.
  - Add the new test described in the G9 section.

## Acceptance

- After an outage run has been saved, an online embedder with the API up returns the API vector, records `stats["api"] == 1`, and serves no hit.
- A cache written by `online=False` with `model=<real model>` is never served to a later online run of the same model.
- Offline cache behaviour is unchanged. These pass unmodified:
  - `tests/test_rag_foundation.py:221-251` (same-instance hit, cross-instance disk hit).
  - `tests/test_rag_ingest.py:147-163` (re-ingest is a no-op, `cache_hits == chunks`). Its embedder is `model="test-embed", online=False` (`:80-87`), which is today's offline-with-model case.
- The online API path is unchanged: `tests/test_rag_foundation.py:105-139` (`test_embedder_routes_to_dedicated_endpoint`) passes unmodified.
- The focused selection passes under the bounded wrapper with `--basetemp`:
  - `tests/test_eval_v2_*.py`
  - `tests/test_rag_foundation.py`, `tests/test_rag_ingest.py`, `tests/test_rag_retrieve.py`, `tests/test_rag_cached_replay.py`
  - If the mining branch has merged by then, the mining suite, which builds its embedder in `tests/helpers_mining.py:441`. Neither exists at `b4bb0b74`.
- SMOKE (508) passes.
- The change to the pinned test in `test_eval_v2_embedding_cache.py` is stated as an exact count delta (merge rule 4).

## G9 obligation

This corrects pre-existing behaviour, so it needs a behavioural proof that fails at `b4bb0b74`.

**Existing proof:** `test_correct_behaviour_online_run_after_an_outage_gets_the_api_vector` (`tests/test_eval_v2_embedding_cache.py:58-63`).

1. `_embed_via_api` is monkeypatched to return `None`; then `embed_one("isochorismate synthase")` and `save()`.
2. A new `Embedder` is built on the same cache path, with `_embed_via_api` returning `[0.5, -0.5, 0.5, -0.5]`.
3. The test asserts that `embed_one` returns that API vector.

At `b4bb0b74` the assertion fails because the lexical vector is served as a hit; it is recorded as XFAIL under the strict marker. At the tip, with the marker removed, it passes.

**New proof, same file:**

1. Run the same sequence, but build the first embedder as `Embedder(model=MODEL, online=False, dim=4, cache_path=tmp_path / "emb.json", config={"embedding_model": MODEL})`.
2. Assert that the online embedder returns `API_VECTOR` and that `stats["api"] == 1`.

At `b4bb0b74` this fails because the key ignores `online` (`embed.py:112`): the online run gets a cache hit with `api == 0`.

**Not proof:** the absence of a model-key entry in `_cache`. Only the served vector and the counters count.

## Risks and interactions

- **Production callers:**
  - `ingest.ingest` (`ingest.py:586`, save at `:623-624`)
  - Streamlit research mode (`streamlit_app.py:578-584`)
  - `embed_texts` (`embed.py:186-198`)
  - eval-v2 `live_retrieval` (`src/t2pw/bench/eval_v2/rag.py:194`)
  - the re-export in `src/t2pw/rag/__init__.py:17`
  - the vector stores, via `embed_fn`
- **Outage cost (inferred, not measured):**
  - During an outage, a repeated text in one process re-attempts the endpoint every time; today it hits the cached fallback.
  - Store queries embed the query text on each call (`store.py:217`, `:356`).
  - A re-ingest during an outage is no longer a no-op (`embedded > 0`).
- **Offline key namespace:** existing offline lexical entries miss once and are recomputed. At the same `dim` the vectors are identical (deterministic, `embed.py:46-58`).
- **SMOKE:**
  - `tests/test_rag_admission_production_path.py` is the only SMOKE file that reaches the embedder. It uses a blank model (`:247-265`), so its vectors are unchanged and only its keys move.
  - `test_eval_v2_embedding_cache.py` is not in the SMOKE list (`TEST_MATRIX.md:259-271`).
- **Mining:**
  - Mining never writes the cache, so the fix changes only what production runs can write. Entries already on disk are still served as hits.
  - Outage misses for repeated texts now count as `lexical` each time.
  - Mining tests use `Embedder(online=False, dim=64)` (`tests/helpers_mining.py:441` at `82fc69a7`), which falls in the lexical namespace and behaves the same.
  - `embed.py` is byte-identical on `b4bb0b74`, `82fc69a7` and `686ae134` (same blob), and `test_eval_v2_embedding_cache.py` is absent from `82fc69a7`, so this card does not conflict with either branch.
- **Not in scope:**
  - `MemoryVectorStore`'s width-mismatch fallback (`store.py:222-235`) stays as a second defence; chroma has none.
  - Dropping old entries by versioning the cache is a separate product-owner call. It is not just a matter of bumping `"version": 1` (`embed.py:171`): `_load_cache` ignores the version (`embed.py:95-109`), so it would also need a version check there, and it would force a full re-embed.

## References

- `src/t2pw/rag/embed.py:41-43, 95-133, 157-183` @ `b4bb0b74`
- `src/t2pw/rag/store.py:222-235, 353-361, 423-446`; `src/t2pw/rag/ingest.py:586, 623-624`; `src/t2pw/app/streamlit_app.py:578-584`; `src/t2pw/config.py:141-142`; `.gitignore:24`
- `tests/test_eval_v2_embedding_cache.py:1-63`; `tests/test_rag_foundation.py:105-139, 221-251`; `tests/test_rag_ingest.py:80-87, 147-163`; `tests/test_rag_admission_production_path.py:247-265`
- `docs/evaluation_v2/README.md:259-266`; `docs/pwml_recovery_sprint/ORCH-742-RESULT.md:126`; `docs/pwml_recovery_sprint/TEST_MATRIX.md:259-271`
- `docs/pwml_recovery_sprint/ORCH-745-RESULT.md:174` (primary checkout); `C:/t/mining-handoff/MINING-OVERNIGHT-HANDOFF.md:101`
- Mining @ `82fc69a7`: `src/t2pw/mining/retrieval.py:305-341`, `preflight.py:153-195, 295-307`, `controller.py:222`, `tests/helpers_mining.py:441`, `docs/pathway_mining/known_limitations.md:42-49`
- Pilot records:
  - `experiment/pathway-mining-pilot-00{1,2,3}:docs/pathway_mining/pilot/00N/run/report.md` (Embedding mode lines)
  - `pilot/00N/pilot-00N.log:1` (resolved project root)
  - `pilot/003/run/preflight.json`
- Primary checkout `data/rag_index/embeddings_cache.json` (untracked; read-only width count for this card)
