"""NEW TESTS (ORCH-742c) -- does the RAG embedder cache keyword/fallback vectors under
the REAL embedding-model key?

FINDING (characterised, not fixed -- production cache behaviour is unchanged):
``t2pw.rag.embed.Embedder.embed_one`` stores whatever ``_embed_uncached`` returns under
``_cache_key(self.model, text)``. When the endpoint is unreachable, that is the
deterministic LEXICAL fallback vector -- so a run made while LM Studio was down writes
bag-of-tokens vectors into the cache under the real model's key, and every later run
(endpoint up) is served those lexical vectors as cache HITS and never re-embeds. When
the configured dimension equals the model's, nothing downstream can tell them apart:
``store.MemoryVectorStore.query`` only falls back to lexical scoring on a WIDTH
mismatch.

The characterisation test pins today's behaviour. The strict-xfail test encodes the
correct behaviour, so it flips loudly (XPASS -> failure) the day the cache is fixed.
Both use a temporary cache path: the repository's data/rag_index cache is never touched,
and no network call is made (the API seam is monkeypatched).
"""

from __future__ import annotations

import pytest

from t2pw.rag import embed as E

MODEL = "text-embedding-real-model"
API_VECTOR = [0.5, -0.5, 0.5, -0.5]


def _embedder(tmp_path, monkeypatch, api_up: bool) -> E.Embedder:
    monkeypatch.setattr(E.Embedder, "_embed_via_api",
                        lambda self, text: list(API_VECTOR) if api_up else None)
    return E.Embedder(model=MODEL, dim=4, cache_path=tmp_path / "emb.json", online=True,
                      config={"embedding_model": MODEL})


def test_characterisation_fallback_vector_is_cached_under_the_real_model_key(tmp_path, monkeypatch):
    down = _embedder(tmp_path, monkeypatch, api_up=False)
    lexical = down.embed_one("isochorismate synthase")
    assert down.stats["lexical"] == 1 and down.stats["api"] == 0
    assert lexical == E._lexical_embedding("isochorismate synthase", 4)
    down.save()

    up = _embedder(tmp_path, monkeypatch, api_up=True)
    assert E._cache_key(MODEL, "isochorismate synthase") in up._cache
    served = up.embed_one("isochorismate synthase")
    # TODAY: the lexical vector comes back as a cache HIT under the real model's key.
    assert served == lexical
    assert up.stats == {"hits": 1, "misses": 0, "api": 0, "lexical": 0}
    assert len(served) == len(API_VECTOR)  # same width, so the store cannot detect it


@pytest.mark.xfail(strict=True, reason=(
    "FINDING ORCH-742c: Embedder caches lexical-fallback vectors under the real embedding-"
    "model key (_cache_key(self.model, text)), so a later online run is served the fallback "
    "vector instead of re-embedding. Correct behaviour: a fallback vector is never cached "
    "under the model key (or is cached under a distinct lexical key)."))
def test_correct_behaviour_online_run_after_an_outage_gets_the_api_vector(tmp_path, monkeypatch):
    down = _embedder(tmp_path, monkeypatch, api_up=False)
    down.embed_one("isochorismate synthase")
    down.save()
    up = _embedder(tmp_path, monkeypatch, api_up=True)
    assert up.embed_one("isochorismate synthase") == API_VECTOR
