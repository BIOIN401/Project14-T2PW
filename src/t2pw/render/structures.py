"""Compound structure pictures for the PathBank-style render (VIS-4).

PathBank's compound boxes show a 2-D structure drawing. We take PubChem's depiction of the compound, by one public
id (only that id leaves the machine), and keep it in an on-disk cache. A key is ``"CHEBI:<n>"`` (PubChem's
cross-reference from ChEBI) or ``"CID:<n>"`` (a PubChem compound id); ``key_for`` prefers ChEBI, because our
PWMLs carry some wrong PubChem CIDs (VIS-4 record) while their ChEBI ids are right.

Cache files, named after the key with ``:`` as ``_``:

- ``<key>.png``: the picture;
- ``<key>.miss``: PubChem answered 404 (or 400 NotFound) for that id; never asked again;
- ``<key>.fail``: anything else that gave no usable picture (network error, timeout, 403 or other status, a body
  that is not a PNG or is over ``MAX_SIDE_PX`` a side or cut at ``MAX_BYTES``), after one retry; asked again after
  ``FAIL_RETRY_S`` (an hour).

The cache directory is ``$T2PW_STRUCTURE_CACHE`` (default ``~/.cache/t2pw/structures``). Requests are paced to at
most 4 per second (PubChem's limit is 5), each has a short timeout, and the whole lookup an overall budget, so a slow
or absent network costs at most ``budget_s`` and the render goes on with empty boxes for what did not arrive.
Nothing here raises.
"""

from __future__ import annotations

import contextlib
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout, as_completed
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

CACHE_ENV = "T2PW_STRUCTURE_CACHE"
BUDGET_ENV = "T2PW_STRUCTURE_BUDGET_S"
DEFAULT_BUDGET_S = 6.0
PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
URLS = {"CID": PUG + "cid/{}/PNG", "CHEBI": PUG + "xref/RegistryID/CHEBI:{}/PNG"}
REQUEST_TIMEOUT_S = 4.0
FAIL_RETRY_S = 3600
WORKERS = 4
MIN_GAP_S = 0.25  # PubChem asks for at most 5 requests per second
RETRY_PAUSE_S = 1.0
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
MAX_BYTES = 2_000_000
MAX_SIDE_PX = 2000  # PubChem's depictions are 300 px; anything far larger is not one (and would decode to GBs)
_ID = re.compile(r"[0-9]{1,10}")  # ASCII digits only: str.isdigit() also accepts "²" and "٣"


class Missing(Exception):
    """PubChem answered 404 (or PUG REST's 400 NotFound): it has no compound for this id."""


class Unusable(Exception):
    """The answer is not a usable PNG (an HTML page, an oversized or truncated image): retried later, never kept."""


def _id(s: str) -> Optional[int]:
    return int(s) if _ID.fullmatch(s) and int(s) > 0 else None


def key_for(chebi_id: str, pubchem_cid: str) -> Optional[str]:
    """The cache and lookup key for a compound: its ChEBI id if it has one, else its PubChem CID, else ``None``.
    Never raises; anything but 1-10 ASCII digits is no id."""
    chebi = _id((chebi_id or "").strip().upper().removeprefix("CHEBI:"))
    if chebi:
        return f"CHEBI:{chebi}"
    cid = _id((pubchem_cid or "").strip())
    return f"CID:{cid}" if cid else None


def _valid(key: str) -> bool:
    kind, _, n = key.partition(":")
    return kind in URLS and _id(n) is not None


def png_ok(data: bytes) -> bool:
    """A PNG whose IHDR says at most ``MAX_SIDE_PX`` a side (read before any decoding)."""
    if len(data) < 24 or not data.startswith(PNG_MAGIC) or data[12:16] != b"IHDR":
        return False
    w, h = int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    return 0 < w <= MAX_SIDE_PX and 0 < h <= MAX_SIDE_PX


def cache_dir() -> Path:
    return Path(os.environ.get(CACHE_ENV) or Path.home() / ".cache" / "t2pw" / "structures")


def default_budget() -> float:
    try:
        return float(os.environ.get(BUDGET_ENV, DEFAULT_BUDGET_S))
    except ValueError:
        return DEFAULT_BUDGET_S


def download(key: str) -> bytes:
    """PubChem's PNG for ``key``. Raises ``Missing`` only for 404 or a 400 PUGREST.NotFound (cached for good);
    ``Unusable`` or the network error for anything else (cached as a retryable failure): a 403 block or a
    maintenance page must not mark every compound missing."""
    kind, _, n = key.partition(":")
    req = urllib.request.Request(URLS[kind].format(n), headers={"User-Agent": "t2pw-render/0.1 (research tool)"})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:  # noqa: S310 - fixed https URL
            data = resp.read(MAX_BYTES + 1)
    except urllib.error.HTTPError as e:
        body = b""
        with contextlib.suppress(Exception):
            body = e.read(4096) if e.fp is not None else b""
        if e.code == 404 or (e.code == 400 and b"NotFound" in body):
            raise Missing(key) from e
        raise
    if len(data) > MAX_BYTES or not png_ok(data):
        raise Unusable(key)
    return data


def _write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except OSError:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


class _Pace:
    """At most one request start per ``MIN_GAP_S`` across the worker threads (PubChem throttles above 5/s with 503)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + MIN_GAP_S
        time.sleep(max(0.0, start - now))


def _get(base: Path, key: str, fetch: Callable[[str], bytes], pace: _Pace, until: float) -> Optional[bytes]:
    for attempt in range(2):  # one retry: a 503 from PubChem's throttle is usually gone a moment later
        pace.wait()
        try:
            data = fetch(key)
        except Missing:
            _write(base.with_suffix(".miss"), b"")
            return None
        except Exception:  # noqa: BLE001 - network trouble of any kind
            if attempt == 0 and time.monotonic() + RETRY_PAUSE_S < until:
                time.sleep(RETRY_PAUSE_S)
                continue
            _write(base.with_suffix(".fail"), b"")  # asked again after FAIL_RETRY_S
            return None
        if not png_ok(data):  # an injected fetcher, or a body cut short: never cached as a picture
            _write(base.with_suffix(".fail"), b"")
            return None
        _write(base.with_suffix(".png"), data)
        return data
    return None


def structures(keys: Iterable[str], budget_s: Optional[float] = None,
               fetch: Optional[Callable[[str], bytes]] = None) -> Dict[str, bytes]:
    """PNG bytes per key (see ``key_for``): from the cache, then fetched (by ``fetch``, default ``download``, looked
    up at call time) within ``budget_s`` seconds (default ``$T2PW_STRUCTURE_BUDGET_S`` or 6; 0 = cache only)."""
    budget_s = default_budget() if budget_s is None else budget_s
    fetch = fetch or download
    d = cache_dir()
    out: Dict[str, bytes] = {}
    todo = []
    now = time.time()
    for key in sorted({k for k in keys if k and _valid(k)}):
        base = d / key.replace(":", "_")
        try:
            out[key] = base.with_suffix(".png").read_bytes()
            continue
        except OSError:
            pass
        if base.with_suffix(".miss").exists():
            continue
        fail = base.with_suffix(".fail")
        if fail.exists() and now - fail.stat().st_mtime < FAIL_RETRY_S:
            continue
        todo.append(key)
    if not todo or budget_s <= 0:
        return out
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        return out
    pool = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="t2pw-structure")
    pace, until = _Pace(), time.monotonic() + budget_s
    futures = {pool.submit(_get, d / key.replace(":", "_"), key, fetch, pace, until): key for key in todo}
    try:
        for f in as_completed(futures, timeout=budget_s):
            data = f.result()
            if data:
                out[futures[f]] = data
    except FuturesTimeout:
        pass  # what is still in flight lands in the cache for the next render
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return out
