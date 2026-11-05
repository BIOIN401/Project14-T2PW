"""Submission limits for the web server (W8): rate limit, upload checks, safe upload saver.

Stand-alone: not wired into ``jobs.py``, ``worker.py`` or any route. W6 calls it, in
this order, before anything is queued::

    key = client_key(remote_addr, request.headers.get("x-forwarded-for"))
    ok, retry_after = RateLimiter(db_path).hit(key)       # rate limit
    problem = check_pathway_count(n) or check_pmids(pmids) or check_text(texts)
    saved = save_uploads(files, upload_dir)               # raises UploadRejected
    # then JobStore.submit(...) with pdf_paths=[s.path for s in saved]

Every user-facing message is a fixed sentence plus, at most, a sanitised display name
and a number. No server path, exception text or raw client file name is ever in one.

Behind a reverse proxy
----------------------
``X-Forwarded-For`` is client-controlled unless a proxy we run rewrites it. So
:func:`client_key` trusts that header only when the TCP peer (``remote_addr``) is in
``T2PW_TRUSTED_PROXIES`` (comma-separated IPs or CIDRs, unset = trust nothing). It then
walks the header right to left and takes the first address that is not itself a trusted
proxy; the entries to its left are client-supplied and ignored. The proxy must append
the peer it saw (nginx ``proxy_add_x_forwarded_for``, not a passthrough of the client's
own header). IPv6 clients are keyed by their /64.

stdlib plus ``pypdf`` (already installed; ``extraction/pdf_parser.py`` uses it).
"""

from __future__ import annotations

import hashlib
import ipaddress
import os
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, List, Optional, Sequence, Tuple

from t2pw.web.jobs import MAX_ITEMS_PER_SUBMISSION  # the max-5-pathways rule lives there

# -- limits (conservative defaults; open question Q-W8-1 in results/W8.md) ----------
#: Submissions per client per rolling window. A run is 20-60 min and the store holds
#: 20 open items, so one client should not be able to fill the queue in a day.
RATE_LIMIT_SUBMISSIONS = 5
RATE_LIMIT_WINDOW_S = 24 * 3600
#: PDFs per submission. A submission holds up to 5 pathways; 10 is two papers each.
MAX_PDF_FILES = 10
#: Typical papers are 0.5-10 MB; 20 MB admits figure-heavy ones and small scans.
MAX_PDF_BYTES = 20 * 1024 * 1024
#: Bounds the disk one submission can take.
MAX_TOTAL_PDF_BYTES = 60 * 1024 * 1024
#: Papers run 8-30 pages, long reviews to ~80. Pages drive OCR and LLM cost.
MAX_PDF_PAGES = 80
#: One pasted-text field is a paper section or a whole paper (~60-100k characters).
MAX_PMIDS = 20
MAX_TEXT_CHARS = 200_000

_CHUNK = 64 * 1024
_MAGIC = b"%PDF-"


class UploadRejected(Exception):
    """An upload failed a check; ``str(exc)`` is the user-facing message."""


# -- client key --------------------------------------------------------------------


def _trusted_networks(spec: Optional[str]) -> list:
    nets = []
    for part in (spec if spec is not None else os.environ.get("T2PW_TRUSTED_PROXIES", "")).split(","):
        part = part.strip()
        if part:
            try:
                nets.append(ipaddress.ip_network(part, strict=False))
            except ValueError:
                pass  # a malformed entry trusts nothing rather than everything
    return nets


def _ip(text: str):
    try:
        return ipaddress.ip_address(text.strip())
    except ValueError:
        return None


def client_key(remote_addr: str, forwarded_for: Optional[str] = None,
               trusted_proxies: Optional[str] = None) -> str:
    """Rate-limit key for a request; see the module docstring for the proxy contract."""
    peer = _ip(remote_addr or "")
    addr = peer
    nets = _trusted_networks(trusted_proxies)
    if peer is not None and forwarded_for and any(peer in n for n in nets):
        for part in reversed(forwarded_for.split(",")):
            ip = _ip(part)
            if ip is None:
                break  # garbage in the chain: stay with the last address we could trust
            addr = ip
            if not any(ip in n for n in nets):
                break
    if addr is None:
        return "unknown"
    if addr.version == 6:
        return str(ipaddress.ip_network(f"{addr}/64", strict=False))
    return str(addr)


# -- rate limit --------------------------------------------------------------------


class RateLimiter:
    """Rolling-window counter in ``rate_events`` of the jobs DB file (created here).

    ``hit(key)`` records one submission attempt unless the key is already at the limit.
    Keys are stored hashed, so the DB holds no client addresses. Each write is one
    ``BEGIN IMMEDIATE`` transaction, like ``jobs.py``, so web and worker share the file.
    """

    def __init__(self, path: Path | str, *, limit: int = RATE_LIMIT_SUBMISSIONS,
                 window_s: float = RATE_LIMIT_WINDOW_S,
                 clock: Callable[[], float] = time.time) -> None:
        self.limit, self.window_s, self._clock = limit, window_s, clock
        self._db = sqlite3.connect(str(path), timeout=30, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(
            "CREATE TABLE IF NOT EXISTS rate_events (key TEXT NOT NULL, ts REAL NOT NULL);"
            "CREATE INDEX IF NOT EXISTS rate_events_key_ts ON rate_events (key, ts);")

    def close(self) -> None:
        self._db.close()

    def hit(self, key: str) -> Tuple[bool, float]:
        """``(True, 0.0)`` and the attempt is counted; or ``(False, seconds_to_wait)``."""
        k = hashlib.sha256(key.encode()).hexdigest()
        now = self._clock()
        self._db.execute("BEGIN IMMEDIATE")
        try:
            # ponytail: prunes every key's expired rows on each hit; fine at this volume.
            self._db.execute("DELETE FROM rate_events WHERE ts <= ?", (now - self.window_s,))
            rows = self._db.execute(
                "SELECT ts FROM rate_events WHERE key = ? ORDER BY ts", (k,)).fetchall()
            if len(rows) >= self.limit:
                # the slot frees when the (len - limit + 1)-th oldest event expires
                oldest_blocking = rows[len(rows) - self.limit][0]
                result = (False, max(0.0, oldest_blocking + self.window_s - now))
            else:
                self._db.execute("INSERT INTO rate_events (key, ts) VALUES (?, ?)", (k, now))
                result = (True, 0.0)
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        self._db.execute("COMMIT")
        return result


def rate_limit_message(retry_after_s: float) -> str:
    minutes = max(1, int(retry_after_s // 60) + 1)
    if minutes >= 120:
        wait = f"{(minutes + 59) // 60} hours"
    else:
        wait = f"{minutes} minutes"
    return ("You have reached the limit of "
            f"{RATE_LIMIT_SUBMISSIONS} submissions per day. Please try again in about {wait}.")


# -- submission-level checks (each: None = fine, else a user-facing message) --------


def check_pathway_count(n: int) -> Optional[str]:
    if n < 1:
        return "Add at least one pathway to the submission."
    if n > MAX_ITEMS_PER_SUBMISSION:
        return f"A submission can hold at most {MAX_ITEMS_PER_SUBMISSION} pathways; you added {n}."
    return None


def check_pmids(pmids: Sequence[str]) -> Optional[str]:
    if len(pmids) > MAX_PMIDS:
        return f"A submission can hold at most {MAX_PMIDS} PubMed IDs; you entered {len(pmids)}."
    return None


def check_text(texts: Sequence[str]) -> Optional[str]:
    total = sum(len(t) for t in texts)
    if total > MAX_TEXT_CHARS:
        return (f"The pasted text is {total:,} characters; the limit is {MAX_TEXT_CHARS:,}. "
                "Paste only the relevant sections, or upload the PDF instead.")
    return None


def check_pdf_count(n: int) -> Optional[str]:
    if n > MAX_PDF_FILES:
        return f"A submission can hold at most {MAX_PDF_FILES} PDF files; you uploaded {n}."
    return None


def check_pdf_bytes(size: int, name: str = "This file") -> Optional[str]:
    if size > MAX_PDF_BYTES:
        return f"{name} is larger than the {MAX_PDF_BYTES // 2**20} MB limit per PDF."
    return None


def check_total_bytes(sizes: Sequence[int]) -> Optional[str]:
    if sum(sizes) > MAX_TOTAL_PDF_BYTES:
        return f"The uploaded PDFs together are larger than the {MAX_TOTAL_PDF_BYTES // 2**20} MB limit."
    return None


def check_pdf_magic(head: bytes, name: str = "This file") -> Optional[str]:
    if not head.startswith(_MAGIC):
        return f"{name} is not a PDF file."
    return None


def count_pdf_pages(path: Path | str) -> Optional[int]:
    """Page count from the PDF's page tree (xref only, no text extraction); ``None``
    when it cannot be read (corrupt, encrypted, truncated)."""
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(path), strict=False).pages)
    except Exception:
        return None


def check_pdf_pages(path: Path | str, name: str = "This file") -> Optional[str]:
    pages = count_pdf_pages(path)
    if pages is None:
        return f"{name} could not be read as a PDF (it may be damaged or password-protected)."
    if pages > MAX_PDF_PAGES:
        return f"{name} has {pages} pages; the limit is {MAX_PDF_PAGES} pages per PDF."
    return None


# -- safe upload saver -------------------------------------------------------------


@dataclass(frozen=True)
class SavedUpload:
    path: Path          # server-owned: <upload_dir>/<random hex>.pdf; safe for pdf_paths
    display_name: str   # sanitised client name, for display only, never a path
    size: int


def sanitize_display_name(name: str, limit: int = 80) -> str:
    """Last path component of a client file name, reduced to ``[\\w .()-]``."""
    base = re.split(r"[\\/]", name or "")[-1]
    base = re.sub(r"[^\w .()\-]", "_", base, flags=re.ASCII).strip(" .")
    return base[:limit] or "upload.pdf"


def save_upload(stream: BinaryIO, upload_dir: Path | str, *, display_name: str = "",
                max_bytes: int = MAX_PDF_BYTES, too_large: Optional[str] = None) -> SavedUpload:
    """Stream ``stream`` into ``upload_dir`` under a random name; run the per-file checks.

    The byte cap is enforced while reading (a client cannot make us buffer or write
    more than ``max_bytes``), the ``%PDF-`` magic is checked on the first chunk, and
    the page cap after the write. On any failure the partial file is removed and
    :class:`UploadRejected` is raised. The client's file name is never used in the path.
    """
    name = sanitize_display_name(display_name)
    folder = Path(upload_dir)
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{uuid.uuid4().hex}.pdf"
    size = 0
    try:
        with dest.open("xb") as out:
            while True:
                chunk = stream.read(_CHUNK)
                if not chunk:
                    break
                if size == 0 and (msg := check_pdf_magic(chunk, name)):
                    raise UploadRejected(msg)
                size += len(chunk)
                if size > max_bytes:
                    raise UploadRejected(too_large or f"{name} is larger than the {max(1, max_bytes // 2**20)} MB limit per PDF.")
                out.write(chunk)
        if size == 0:
            raise UploadRejected(f"{name} is empty.")
        if msg := check_pdf_pages(dest, name):
            raise UploadRejected(msg)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return SavedUpload(dest, name, size)


def save_uploads(files: Sequence[Tuple[str, BinaryIO]], upload_dir: Path | str) -> List[SavedUpload]:
    """Save ``(client_name, stream)`` pairs, all or nothing, under the count and total caps."""
    if msg := check_pdf_count(len(files)):
        raise UploadRejected(msg)
    saved: List[SavedUpload] = []
    try:
        for name, stream in files:
            remaining = MAX_TOTAL_PDF_BYTES - sum(s.size for s in saved)
            total_cap = remaining < MAX_PDF_BYTES
            saved.append(save_upload(
                stream, upload_dir, display_name=name, max_bytes=min(MAX_PDF_BYTES, remaining),
                too_large=check_total_bytes([MAX_TOTAL_PDF_BYTES + 1]) if total_cap else None))
    except BaseException:
        for s in saved:
            s.path.unlink(missing_ok=True)
        raise
    return saved
