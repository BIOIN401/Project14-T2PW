"""Submission limits for the web server (W8): rate limit, upload checks, safe upload saver.

Stand-alone: not wired into ``jobs.py``, ``worker.py`` or any route. W6 calls it, in
this order, before anything is queued: cheap validation first, so a typo does not burn a
daily slot; then the rate limit; then the (costly) upload save::

    problem = check_pathway_count(n) or check_pmids(pmids) or check_text(texts)
    key = client_key(remote_addr, request.headers.get("x-forwarded-for"))
    ok, retry_after = RateLimiter(db_path).hit(key)       # counts the attempt
    saved = save_uploads(files, upload_dir)               # raises UploadRejected
    # then JobStore.submit(...) with pdf_paths=[s.path for s in saved]

An upload rejected inside ``save_uploads`` has already used its slot; that is deliberate
(it stops a client probing the PDF checks for free).

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
own header). IPv6 clients are keyed by their /64; IPv4-mapped IPv6 (``::ffff:a.b.c.d``,
what a dual-stack listener reports) is treated as the IPv4 address it wraps.

Page counting parses an untrusted PDF, which can be slow or hostile, so it runs in a
short-lived child interpreter with a hard timeout (:data:`PAGE_COUNT_TIMEOUT_S`) that is
killed on expiry; encrypted PDFs are refused (pypdf reports the attacker-declared
``/Count`` for them). stdlib plus ``pypdf`` (already installed; ``extraction/pdf_parser.py``
uses it).
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import logging
import os
import re
import secrets
import sqlite3
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, List, Optional, Sequence, Tuple

log = logging.getLogger("t2pw.web.limits")

from t2pw.web.jobs import MAX_ITEMS_PER_SUBMISSION  # the max-5-pathways rule lives there

# -- limits (conservative defaults; open question Q-W8-1 in results/W8.md) ----------
#: Submissions per client per rolling window. A run is 20-60 min (stopped at 90) and the store holds
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

#: Wall-clock cap for the child process that counts a PDF's pages.
PAGE_COUNT_TIMEOUT_S = 10

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
                net = ipaddress.ip_network(part, strict=False)
            except ValueError:
                continue  # a malformed entry trusts nothing rather than everything
            mapped = net.network_address.ipv4_mapped if net.version == 6 else None
            if mapped is not None and net.prefixlen >= 96:  # ::ffff:10.0.0.0/104 -> 10.0.0.0/8
                net = ipaddress.ip_network(f"{mapped}/{net.prefixlen - 96}", strict=False)
            nets.append(net)
    return nets


def _ip(text: str):
    """Parse an address; an IPv4-mapped IPv6 address becomes the IPv4 address it wraps."""
    try:
        ip = ipaddress.ip_address(text.strip())
    except ValueError:
        return None
    return ip.ipv4_mapped or ip if ip.version == 6 else ip


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
    Keys are stored as an HMAC-SHA256 under a server secret (``secret``, else
    ``T2PW_RATE_KEY_SECRET``), not as addresses: a leaked DB does not list client IPs, and
    without the secret the small IPv4 space cannot be brute-forced back out of it. With no
    secret configured a random per-process one is used (logged as a warning): counts then
    do not survive a restart and are not shared between processes. Each write is one
    ``BEGIN IMMEDIATE`` transaction, like ``jobs.py``, so web and worker share the file.
    """

    def __init__(self, path: Path | str, *, limit: int = RATE_LIMIT_SUBMISSIONS,
                 window_s: float = RATE_LIMIT_WINDOW_S,
                 clock: Callable[[], float] = time.time,
                 secret: Optional[bytes] = None) -> None:
        self.limit, self.window_s, self._clock = limit, window_s, clock
        if secret is None and os.environ.get("T2PW_RATE_KEY_SECRET"):
            secret = os.environ["T2PW_RATE_KEY_SECRET"].encode()
        if secret is None:
            log.warning("T2PW_RATE_KEY_SECRET is not set: using a random per-process rate-limit "
                        "secret, so limits reset on restart and are not shared between processes")
            secret = secrets.token_bytes(32)
        self._secret = secret
        self._db = sqlite3.connect(str(path), timeout=30, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(
            "CREATE TABLE IF NOT EXISTS rate_events (key TEXT NOT NULL, ts REAL NOT NULL);"
            "CREATE INDEX IF NOT EXISTS rate_events_key_ts ON rate_events (key, ts);")

    def close(self) -> None:
        self._db.close()

    def hit(self, key: str) -> Tuple[bool, float]:
        """``(True, 0.0)`` and the attempt is counted; or ``(False, seconds_to_wait)``."""
        k = hmac.new(self._secret, key.encode(), hashlib.sha256).hexdigest()
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


#: Runs in the child interpreter. Prints ``ENCRYPTED`` or the page count; any failure is a
#: non-zero exit. ``pages`` makes pypdf walk the whole page tree (no text extraction),
#: which is why this is not run in the web process.
_COUNT_CODE = (
    "import sys\n"
    "from pypdf import PdfReader\n"
    "r = PdfReader(sys.argv[1], strict=False)\n"
    "print('ENCRYPTED' if r.is_encrypted else len(r.pages))\n"
)


def count_pdf_pages(path: Path | str, name: str = "This file", *,
                    timeout_s: Optional[float] = None, child_code: str = _COUNT_CODE) -> int:
    """Number of pages in the PDF at ``path``, counted in a child process.

    pypdf's page count walks the whole page tree of an untrusted file (unbounded work),
    and for an encrypted file returns the attacker-declared ``/Count``. So the count runs
    in a short-lived ``python -I -c`` child with a hard timeout
    (:data:`PAGE_COUNT_TIMEOUT_S`); ``subprocess.run`` kills that one child on expiry.
    Raises :class:`UploadRejected` (safe message) for an encrypted, unreadable or
    too-slow file. ``child_code`` exists so tests can inject a hanging child.
    """
    timeout = PAGE_COUNT_TIMEOUT_S if timeout_s is None else timeout_s
    try:
        done = subprocess.run([sys.executable, "-I", "-c", child_code, str(path)],
                              stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise UploadRejected(f"{name} could not be read as a PDF (it took too long to open).") from None
    except OSError:
        raise UploadRejected(f"{name} could not be checked right now; please try again later.") from None
    out = done.stdout.decode("ascii", "replace").strip()
    if done.returncode == 0 and out == "ENCRYPTED":
        raise UploadRejected(f"{name} is encrypted. Encrypted PDFs are not supported; "
                             "please upload an unencrypted copy.")
    if done.returncode == 0 and out.isdigit():
        return int(out)
    raise UploadRejected(f"{name} could not be read as a PDF (it may be damaged).")


def check_pdf_pages(path: Path | str, name: str = "This file", **kw) -> Optional[str]:
    try:
        pages = count_pdf_pages(path, name, **kw)
    except UploadRejected as exc:
        return str(exc)
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
    more than ``max_bytes``), the ``%PDF-`` magic is checked as soon as the first five
    bytes have arrived (however the stream chunks them), and the page cap (and encryption
    check) after the write, in a bounded child process. On any failure the partial file is
    removed and :class:`UploadRejected` is raised. The client's file name is never used in
    the path. ``upload_dir`` is created ``0o700`` and a symlinked one is refused; files are
    created ``0o600`` and exclusively (``O_EXCL`` also refuses a pre-planted symlink).
    """
    name = sanitize_display_name(display_name)
    folder = Path(upload_dir)
    if folder.is_symlink() or getattr(os.path, "isjunction", lambda p: False)(folder):
        log.error("upload directory is a link; refusing to save")
        raise UploadRejected("Uploads are not available right now; please try again later.")
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    dest = folder / f"{uuid.uuid4().hex}.pdf"
    size, head = 0, b""
    try:
        fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = stream.read(_CHUNK)
                if not chunk:
                    break
                if len(head) < len(_MAGIC):
                    head += chunk[:len(_MAGIC) - len(head)]
                    if not _MAGIC.startswith(head):
                        raise UploadRejected(check_pdf_magic(head, name))
                size += len(chunk)
                if size > max_bytes:
                    raise UploadRejected(too_large or f"{name} is larger than the {max(1, max_bytes // 2**20)} MB limit per PDF.")
                out.write(chunk)
        if size == 0:
            raise UploadRejected(f"{name} is empty.")
        if len(head) < len(_MAGIC):
            raise UploadRejected(check_pdf_magic(head, name))
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
