"""Embedding client for the RAG subsystem (WP0).

Produces embedding vectors for chunk / query text using the existing
OpenAI-compatible client (``t2pw.llm.client``) against an embeddings endpoint,
with two safety nets that mirror the rest of the pipeline's offline-first design:

* **Cache** — vectors are cached in ``data/rag_index/embeddings_cache.json``
  (mirroring ``data/id_mapping_cache.json``); an unchanged chunk is never
  re-embedded.
* **Lexical offline fallback** — if no embedding endpoint is reachable (no key,
  no network, unsupported model), the embedder deterministically derives a
  bag-of-tokens vector so retrieval still returns *something*. RAG never
  hard-fails on a missing embedder.

``t2pw.llm.client`` is imported lazily inside the API call so that merely
importing this module (or ``t2pw.rag``) never constructs the LLM client or
requires an API key.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from t2pw.config import rag_config
from t2pw.paths import PROJECT_ROOT

_DEFAULT_LEXICAL_DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return _TOKEN_RE.findall(text.casefold())


def _cache_key(model: str, text: str) -> str:
    digest = hashlib.sha256(f"{model}\x00{text}".encode("utf-8")).hexdigest()
    return digest


def _lexical_embedding(text: str, dim: int) -> List[float]:
    """Deterministic, offline bag-of-tokens vector (hashing trick, L2-normed)."""
    dim = max(1, int(dim))
    vec = [0.0] * dim
    for token in _tokenize(text):
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0.0:
        vec = [v / norm for v in vec]
    return vec


class Embedder:
    """Cached, offline-capable embedding client.

    Set ``online=False`` (or leave the endpoint unreachable) to force the
    deterministic lexical fallback — useful for tests and for fully offline runs.
    ``stats`` exposes cache ``hits`` / ``misses`` and how each miss was resolved
    (``api`` vs ``lexical``).
    """

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        dim: Optional[int] = None,
        cache_path: Optional[Path] = None,
        online: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = config or rag_config()
        self.model = (model if model is not None else str(cfg.get("embedding_model", ""))).strip()
        configured_dim = int(cfg.get("embedding_dim", 0) or 0)
        self.dim = int(dim) if dim else (configured_dim or _DEFAULT_LEXICAL_DIM)
        self.online = bool(online)
        # Dedicated embeddings endpoint (e.g. LM Studio / OpenAI). Blank -> reuse
        # the shared LLM client. This is what lets embeddings target a different
        # host than chat (OpenRouter serves chat but not embeddings).
        self.base_url = str(cfg.get("embedding_base_url", "")).strip()
        self.api_key = str(cfg.get("embedding_api_key", "")).strip()
        self._api_client: Any = None  # constructed lazily on first API embed
        self.cache_path = Path(cache_path) if cache_path else default_cache_path(cfg)
        self._cache: Dict[str, List[float]] = self._load_cache()
        self.stats: Dict[str, int] = {"hits": 0, "misses": 0, "api": 0, "lexical": 0}
        self._dirty = False

    def _load_cache(self) -> Dict[str, List[float]]:
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - missing/broken cache is not an error
            return {}
        if not isinstance(data, dict):
            return {}
        entries = data.get("entries", data)
        if not isinstance(entries, dict):
            return {}
        out: Dict[str, List[float]] = {}
        for key, vec in entries.items():
            if isinstance(vec, list):
                out[str(key)] = [float(x) for x in vec]
        return out

    def embed_one(self, text: str) -> List[float]:
        key = _cache_key(self.model, text)
        cached = self._cache.get(key)
        if cached is not None:
            self.stats["hits"] += 1
            return cached
        self.stats["misses"] += 1
        vec = self._embed_uncached(text)
        self._cache[key] = vec
        self._dirty = True
        return vec

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_one(t) for t in texts]

    def _embed_uncached(self, text: str) -> List[float]:
        if self.online:
            vec = self._embed_via_api(text)
            if vec is not None:
                self.stats["api"] += 1
                return vec
        self.stats["lexical"] += 1
        return _lexical_embedding(text, self.dim)

    def _get_api_client(self) -> Any:
        """Return the OpenAI-compatible client used for embeddings.

        When ``embedding_base_url`` is configured (e.g. LM Studio at
        ``http://127.0.0.1:1234/v1``), build a dedicated client pointed there —
        this is required when chat and embeddings live on different hosts
        (OpenRouter serves chat but has no embeddings endpoint). Otherwise reuse
        the shared LLM client, preserving the original behavior.
        """
        if self._api_client is not None:
            return self._api_client
        if self.base_url:
            from openai import OpenAI  # lazy: never import at module load

            # LM Studio ignores the key, but the SDK requires a non-empty string.
            self._api_client = OpenAI(base_url=self.base_url, api_key=(self.api_key or "not-needed"))
        else:
            from t2pw.llm import client as llm_client

            self._api_client = llm_client._client
        return self._api_client

    def _embed_via_api(self, text: str) -> Optional[List[float]]:
        if not self.model:
            return None
        try:
            client = self._get_api_client()
            resp = client.embeddings.create(model=self.model, input=text)
            vector = list(resp.data[0].embedding)
            return [float(x) for x in vector]
        except Exception:  # noqa: BLE001 - any failure -> offline lexical fallback
            return None

    def save(self) -> None:
        if not self._dirty:
            return
        payload = {"version": 1, "model": self.model, "entries": self._cache}
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._dirty = False


def default_cache_path(config: Optional[Dict[str, Any]] = None) -> Path:
    cfg = config or rag_config()
    raw = Path(str(cfg.get("index_dir", "data/rag_index")))
    base = raw if raw.is_absolute() else (PROJECT_ROOT / raw)
    return base / "embeddings_cache.json"


def embed_texts(
    texts: List[str],
    *,
    model: Optional[str] = None,
    online: bool = True,
    save: bool = True,
) -> List[List[float]]:
    """Convenience one-shot embed that loads, uses, and (optionally) saves cache."""
    embedder = Embedder(model=model, online=online)
    vectors = embedder.embed(texts)
    if save:
        embedder.save()
    return vectors


__all__ = ["Embedder", "embed_texts", "default_cache_path"]
