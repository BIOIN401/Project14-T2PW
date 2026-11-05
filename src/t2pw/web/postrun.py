"""Post-final actions for the worker (W7): curator enqueue and the one job email.

:func:`run` is called by the worker each time an item settles. It acts only on an item
that is FINAL (``finished``): an item re-queued for a retry does nothing. Then:

1. curator enqueue, if the item qualifies (:mod:`t2pw.web.curator_queue`);
2. when every item of the submission is final and it has an email, one
   :func:`t2pw.web.notify.notify_job` call.

Exactly once for the email: a row in ``web_notifications`` (primary key = submission id)
is claimed in a transaction BEFORE sending, so a second call, a second item or a worker
restart finds the claim and does nothing. The price is at-most-once: a crash between the
claim and the send loses that email (the row stays ``sending``), and a failed send is not
retried (``not_sent``). Nothing here raises, and nothing changes the item's result: a
failure is logged as its exception class and ids only (no address, no secret).
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from t2pw.service.pathway import STATUS_PASS
from t2pw.web.curator_queue import CuratorQueue
from t2pw.web.jobs import FINISHED, JobStore
from t2pw.web.notify import notify_job

log = logging.getLogger("t2pw.web.postrun")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS web_notifications (
    submission_id TEXT PRIMARY KEY,
    outcome       TEXT NOT NULL,
    state         TEXT NOT NULL CHECK (state IN ('sending', 'sent', 'not_sent')),
    claimed_at    REAL NOT NULL,
    done_at       REAL
);
"""


@dataclass
class PostRunConfig:
    queue_dir: Path
    work_root: Optional[Path] = None        # curator copies must come from under here
    base_url: Optional[str] = None          # no base_url -> no email (no safe link)
    notify: Callable[..., bool] = notify_job

    @classmethod
    def from_env(cls, db_path: Path | str, work_root: Optional[Path] = None,
                 env: Optional[Dict[str, str]] = None) -> "PostRunConfig":
        env = os.environ if env is None else env
        queue = env.get("T2PW_CURATOR_QUEUE_DIR") or Path(db_path).parent / "curator_queue"
        return cls(queue_dir=Path(queue), work_root=work_root,
                   base_url=(env.get("T2PW_BASE_URL") or "").strip() or None)


def _claim(db_path: Path | str, sid: str, outcome: str) -> bool:
    db = sqlite3.connect(str(db_path), timeout=30, isolation_level=None)
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript(_SCHEMA)
        cur = db.execute("INSERT OR IGNORE INTO web_notifications "
                         "(submission_id, outcome, state, claimed_at) VALUES (?, ?, 'sending', ?)",
                         (sid, outcome, time.time()))
        return cur.rowcount == 1
    finally:
        db.close()


def _done(db_path: Path | str, sid: str, sent: bool) -> None:
    db = sqlite3.connect(str(db_path), timeout=30, isolation_level=None)
    try:
        db.execute("UPDATE web_notifications SET state = ?, done_at = ? WHERE submission_id = ?",
                   ("sent" if sent else "not_sent", time.time(), sid))
    finally:
        db.close()


def run(store: JobStore, item_id: int, cfg: PostRunConfig) -> Dict[str, Any]:
    """Run the post-final actions for ``item_id``; never raises. Returns what happened
    (``enqueued``: bool, ``notified``: None / True / False) for tests and logs."""

    out: Dict[str, Any] = {"enqueued": False, "notified": None}
    try:
        item = store.item(item_id)
        if item is None or item["state"] != FINISHED:
            return out
        sub = store.submission(item["submission_id"])
    except Exception as exc:  # noqa: BLE001
        log.warning("post-final read failed for item %s: %s", item_id, type(exc).__name__)
        return out

    try:
        q = CuratorQueue(store.path, cfg.queue_dir)
        try:
            out["enqueued"] = q.enqueue(sub, item, cfg.work_root) is not None
        finally:
            q.close()
    except Exception as exc:  # noqa: BLE001
        log.warning("curator enqueue failed for item %s: %s", item_id, type(exc).__name__)

    try:
        items = sub["items"]
        if sub["email"] and cfg.base_url and all(i["state"] == FINISHED for i in items):
            outcome = "finished" if any(i["status"] == STATUS_PASS for i in items) else "failed"
            sid = sub["id"]
            if _claim(store.path, sid, outcome):
                try:
                    sent = bool(cfg.notify(sub["email"], f"{cfg.base_url.rstrip('/')}/jobs/{sid}",
                                           outcome=outcome, base_url=cfg.base_url))
                except Exception as exc:  # noqa: BLE001
                    log.warning("job email failed for submission %s: %s", sid, type(exc).__name__)
                    sent = False
                _done(store.path, sid, sent)
                out["notified"] = sent
    except Exception as exc:  # noqa: BLE001
        log.warning("job email step failed for item %s: %s", item_id, type(exc).__name__)
    return out


def sweep(store: JobStore, cfg: PostRunConfig) -> int:
    """Start-up catch-up for a crash between ``store.finish`` and :func:`run`: curator
    enqueue ONLY (idempotent) for finished ``pass`` items with no queue row. It never sends
    email, so history is not mailed. Never raises; returns how many items it enqueued."""

    n = 0
    try:
        q = CuratorQueue(store.path, cfg.queue_dir)
        try:
            ids = [r[0] for r in q._db.execute(
                "SELECT i.id FROM items i LEFT JOIN curator_queue c ON c.item_id = i.id "
                "WHERE i.state = 'finished' AND i.status = ? AND c.item_id IS NULL ORDER BY i.id",
                (STATUS_PASS,))]
            for iid in ids:
                try:
                    item = store.item(iid)
                    n += q.enqueue(store.submission(item["submission_id"]), item, cfg.work_root) is not None
                except Exception as exc:  # noqa: BLE001
                    log.warning("curator sweep failed for item %s: %s", iid, type(exc).__name__)
        finally:
            q.close()
    except Exception as exc:  # noqa: BLE001
        log.warning("curator sweep failed: %s", type(exc).__name__)
    return n
