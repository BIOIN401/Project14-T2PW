"""On-disk JSON cache keyed by request (url + params). Negative results are cached; every entry stores its retrieval date."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional, Tuple

DEFAULT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "clean_model_cache"))


def request_key(kind: str, url: str, params: Optional[Dict[str, Any]]) -> str:
    raw = json.dumps([kind, url, sorted((params or {}).items())], sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class Cache:
    """``folder=None`` keeps entries in memory only (tests). Never touches data/*_cache.json."""

    def __init__(self, folder: Optional[str] = DEFAULT_DIR) -> None:
        self.folder = folder
        self._mem: Dict[str, Dict[str, Any]] = {}

    def _path(self, key: str) -> Optional[str]:
        return os.path.join(self.folder, key[:2], key + ".json") if self.folder else None

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if key in self._mem:
            return self._mem[key]
        p = self._path(key)
        if p and os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as fh:
                    self._mem[key] = json.load(fh)
                    return self._mem[key]
            except (OSError, ValueError):
                return None
        return None

    def put(self, key: str, url: str, params: Optional[Dict[str, Any]], value: Any, retrieved: str) -> None:
        entry = {"url": url, "params": params or {}, "retrieved": retrieved, "value": value}  # value None = negative
        self._mem[key] = entry
        p = self._path(key)
        if p:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(entry, fh, ensure_ascii=False)


class CachedFetcher:
    """Wraps a Fetcher with the cache and counts live requests vs cache hits (a FetchError is not cached)."""

    def __init__(self, inner: Any, cache: Cache, today: str) -> None:
        self.inner, self.cache, self.today = inner, cache, today
        self.requests = 0
        self.hits = 0
        self.dates: list = []  # retrieval date of every response served since the caller last cleared it

    def _go(self, kind: str, url: str, params: Optional[Dict[str, Any]]) -> Any:
        key = request_key(kind, url, params)
        hit = self.cache.get(key)
        if hit is not None:
            self.hits += 1
            self.dates.append(hit.get("retrieved") or self.today)  # the entry's own date, not today's
            return hit["value"]
        self.requests += 1
        value = getattr(self.inner, "get_" + kind)(url, params)
        self.cache.put(key, url, params, value, self.today)
        self.dates.append(self.today)
        return value

    def get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._go("json", url, params)

    def get_text(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[str]:
        return self._go("text", url, params)
