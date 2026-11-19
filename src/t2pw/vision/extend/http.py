"""One cached, deadline-aware GET for the public REST sources (Reactome, WikiPathways)."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Optional

USER_AGENT = "t2pw-vision/0.1 (research tool)"


class SourceError(RuntimeError):
    """A source cannot answer; the message is ours and safe to record."""


class BudgetExceeded(SourceError):
    pass


def get_text(url: str, *, deadline: float, cache_dir=None, timeout: float = 30.0) -> Optional[str]:
    """Body of ``url``; None on 404 (the services answer 'no mapping' that way). Responses are cached
    by URL under ``cache_dir`` (no expiry: delete the directory to refresh). Raises BudgetExceeded when
    the deadline (time.monotonic) has passed, and requests' errors on any other failure."""
    path = Path(cache_dir) / (hashlib.sha256(url.encode()).hexdigest()[:32] + ".txt") if cache_dir else None
    if path and path.exists():
        body = path.read_text(encoding="utf-8")
        return None if body == "\0404" else body
    left = deadline - time.monotonic()
    if left <= 1:
        raise BudgetExceeded(url)
    import requests  # installed; imported here so the package imports without it

    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=min(timeout, left))
    if r.status_code == 404:
        body = None
    else:
        r.raise_for_status()
        body = r.text
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\0404" if body is None else body, encoding="utf-8")
    return body
