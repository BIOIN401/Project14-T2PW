"""HTTP fetching: stdlib urllib, generic User-Agent (no email), per-host spacing, retries with backoff."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, Optional, Protocol, Tuple

USER_AGENT = "t2pw-clean-model/0.1 (research tool)"
MIN_INTERVAL = 0.34  # seconds: at most 3 requests per second per host (UniProt, EBI, NCBI eutils without a key, Rhea)
RETRY_STATUS = (429, 500, 502, 503, 504)


class FetchError(Exception):
    """A request failed after its retries (network error, timeout, or persistent 429/5xx)."""


class Fetcher(Protocol):
    def get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any: ...
    def get_text(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[str]: ...


class Throttle:
    """Keeps ``interval`` seconds between requests to the same host. Clock and sleep are injectable."""

    def __init__(self, interval: float = MIN_INTERVAL, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.interval, self.clock, self.sleep = interval, clock, sleep
        self._last: Dict[str, float] = {}

    def wait(self, host: str) -> float:
        now = self.clock()
        last = self._last.get(host)
        delay = 0.0 if last is None else max(0.0, last + self.interval - now)
        if delay:
            self.sleep(delay)
        self._last[host] = now + delay
        return delay


def _urlopen(url: str, headers: Dict[str, str], timeout: float) -> Tuple[int, Dict[str, str], bytes]:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (https URLs only, built by sources.py)
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), e.read() or b""


class UrllibFetcher:
    """404 returns None (a definite "not found"); 429/5xx are retried, then raise FetchError."""

    def __init__(self, throttle: Optional[Throttle] = None, retries: int = 3, backoff: float = 1.0,
                 timeout: float = 30.0, opener: Callable[..., Tuple[int, Dict[str, str], bytes]] = _urlopen,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.throttle, self.retries, self.backoff = throttle or Throttle(), retries, backoff
        self.timeout, self._open, self._sleep = timeout, opener, sleep

    def _get(self, url: str, params: Optional[Dict[str, Any]], accept: str) -> Optional[bytes]:
        if params:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
        host = urllib.parse.urlsplit(url).netloc
        headers = {"User-Agent": USER_AGENT, "Accept": accept}
        status = 0
        for attempt in range(self.retries + 1):
            self.throttle.wait(host)
            try:
                status, hdrs, body = self._open(url, headers, self.timeout)
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                status, hdrs, body = -1, {}, str(e).encode()
            if status == 200:
                return body
            if status == 404:
                return None
            if status != -1 and status not in RETRY_STATUS:
                raise FetchError(f"HTTP {status} for {url}")
            if attempt < self.retries:
                try:
                    wait = float(hdrs.get("Retry-After", ""))
                except ValueError:
                    wait = self.backoff * 2 ** attempt
                self._sleep(min(wait, 60.0))
        raise FetchError(f"HTTP {status} after {self.retries + 1} tries for {url}")

    def get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        body = self._get(url, params, "application/json")
        try:  # a 200 with an HTML error page or a truncated body must not abort the whole fill
            return None if body is None else json.loads(body.decode("utf-8"))
        except ValueError as e:  # JSONDecodeError and UnicodeDecodeError
            raise FetchError(f"HTTP 200 but the body is not JSON for {url}: {e}") from e

    def get_text(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[str]:
        body = self._get(url, params, "text/plain, text/tab-separated-values")
        try:
            return None if body is None else body.decode("utf-8")
        except ValueError as e:
            raise FetchError(f"HTTP 200 but the body is not UTF-8 text for {url}: {e}") from e
