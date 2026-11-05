"""SQLite job store for the web server (W3).

A *submission* holds 1 to :data:`MAX_ITEMS_PER_SUBMISSION` *items*; each item is one
:class:`~t2pw.service.pathway.PathwayRequest`. The store also keeps the submission's
PathBank consent flag (default on) and optional email. Those two are stored only:
W7 and W8 act on them.

Item states: ``queued`` -> ``running`` -> ``finished``. ``finished`` says the item is
no longer being worked on; how it ended is the persisted result's ``status`` and
``failure_kind`` (the ``t2pw.service.pathway`` vocabulary). Every claim bumps the
item's ``attempt`` and opens one ``attempts`` row, so W4 can re-queue a finished item
(``state='queued'``) without a schema change. W3 itself never retries.

stdlib only. Every write is one ``BEGIN IMMEDIATE`` transaction, so the web process
and the worker process can share the file; the queue-cap check and the insert it
guards are in the same transaction.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from t2pw.rag.eligibility import RequestedScope
from t2pw.service.pathway import (
    KIND_CRASH,
    STAGE_INIT,
    STATUS_ERROR,
    PathwayRequest,
    PathwayResult,
)

MAX_ITEMS_PER_SUBMISSION = 5

#: Total open items (queued + running) the store accepts. A run takes 20-60 min on one
#: worker, so 20 items is roughly 7-20 hours of backlog. Open question Q-W3-1.
MAX_QUEUED_ITEMS = 20

QUEUED, RUNNING, FINISHED = "queued", "running", "finished"

RECOVERED_MESSAGE = (
    "the worker stopped while this item was running (attempt {attempt}); "
    "marked as a crash when the worker started again"
)


class QueueFull(Exception):
    """The submission would take the open-item count past :data:`MAX_QUEUED_ITEMS`."""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS submissions (
    id         TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    consent    INTEGER NOT NULL DEFAULT 1 CHECK (consent IN (0, 1)),
    email      TEXT
);
CREATE TABLE IF NOT EXISTS items (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id TEXT NOT NULL REFERENCES submissions(id),
    position      INTEGER NOT NULL,
    request_json  TEXT NOT NULL,
    state         TEXT NOT NULL DEFAULT 'queued'
                  CHECK (state IN ('queued', 'running', 'finished')),
    attempt       INTEGER NOT NULL DEFAULT 0,
    deadline      REAL,
    status        TEXT,
    failure_kind  TEXT,
    result_json   TEXT,
    created_at    REAL NOT NULL,
    started_at    REAL,
    finished_at   REAL
);
CREATE TABLE IF NOT EXISTS attempts (
    item_id      INTEGER NOT NULL REFERENCES items(id),
    attempt      INTEGER NOT NULL,
    started_at   REAL NOT NULL,
    deadline     REAL,
    finished_at  REAL,
    child_pid    INTEGER,
    exit_code    INTEGER,
    status       TEXT,
    failure_kind TEXT,
    message      TEXT,
    PRIMARY KEY (item_id, attempt)
);
"""


# ── (de)serialisation ──────────────────────────────────────────────────────


def request_to_dict(req: PathwayRequest) -> Dict[str, Any]:
    return {
        "texts": list(req.texts),
        "pdf_paths": [str(p) for p in req.pdf_paths],
        "pmids": list(req.pmids),
        "scope": req.scope.to_dict(),
        "extraction_focus": req.extraction_focus,
        "export_mode": req.export_mode,
        "out_dir": None if req.out_dir is None else str(req.out_dir),
        "deadline": req.deadline,
    }


def request_from_dict(d: Dict[str, Any]) -> PathwayRequest:
    return PathwayRequest(
        texts=tuple(d["texts"]),
        pdf_paths=tuple(Path(p) for p in d["pdf_paths"]),
        pmids=tuple(d["pmids"]),
        scope=RequestedScope(**d["scope"]),
        extraction_focus=d["extraction_focus"],
        export_mode=d["export_mode"],
        out_dir=None if d["out_dir"] is None else Path(d["out_dir"]),
        deadline=d["deadline"],
    )


def result_to_dict(res: PathwayResult) -> Dict[str, Any]:
    """JSON-safe copy of every ``PathwayResult`` field; ``release_status`` via ``to_dict``."""

    def path(p: Optional[Path]) -> Optional[str]:
        return None if p is None else str(p)

    return {
        "status": res.status,
        "stage": res.stage,
        "failure_kind": res.failure_kind,
        "message": res.message,
        "warnings": list(res.warnings),
        "issue_codes": list(res.issue_codes),
        "counts": dict(res.counts),
        "release_status": None if res.release_status is None else res.release_status.to_dict(),
        "pwml_path": path(res.pwml_path),
        "image_path": path(res.image_path),
        "report_paths": {k: str(v) for k, v in res.report_paths.items()},
        "attempt_log": list(res.attempt_log),
    }


# ── the store ──────────────────────────────────────────────────────────────


class JobStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._db = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield self._db
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        self._db.execute("COMMIT")

    # writes

    def submit(self, requests: Sequence[PathwayRequest], *, consent: bool = True,
               email: Optional[str] = None) -> str:
        """Queue one submission; returns its id. All or nothing.

        Raises ``ValueError`` for 0 or more than 5 items, ``TypeError`` for a
        non-``PathwayRequest`` item, :class:`QueueFull` past the cap. A ``deadline``
        on a submitted request is ignored: the worker sets it at dequeue.
        """

        requests = list(requests)
        if not 1 <= len(requests) <= MAX_ITEMS_PER_SUBMISSION:
            raise ValueError(
                f"a submission holds 1 to {MAX_ITEMS_PER_SUBMISSION} pathways, got {len(requests)}")
        if not all(isinstance(r, PathwayRequest) for r in requests):
            raise TypeError("every item must be a PathwayRequest")
        sid, now = uuid.uuid4().hex, time.time()
        with self._tx() as db:
            (open_items,) = db.execute(
                "SELECT COUNT(*) FROM items WHERE state IN ('queued', 'running')").fetchone()
            if open_items + len(requests) > MAX_QUEUED_ITEMS:
                raise QueueFull(
                    f"queue holds {open_items} of {MAX_QUEUED_ITEMS} pathways; "
                    f"{len(requests)} more do not fit")
            db.execute("INSERT INTO submissions (id, created_at, consent, email) VALUES (?, ?, ?, ?)",
                       (sid, now, int(bool(consent)), email or None))
            db.executemany(
                "INSERT INTO items (submission_id, position, request_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                [(sid, i, json.dumps(request_to_dict(r)), now) for i, r in enumerate(requests)])
        return sid

    def claim_next(self, timeout_s: float) -> Optional[Dict[str, Any]]:
        """Oldest queued item -> ``running``, next attempt, deadline = now + ``timeout_s``.

        The deadline is set here, at dequeue, so time spent queued never eats the
        run's wall-clock cap.
        """

        with self._tx() as db:
            row = db.execute(
                "SELECT id, attempt FROM items WHERE state = 'queued' ORDER BY id LIMIT 1").fetchone()
            if row is None:
                return None
            now = time.time()
            attempt, deadline = row["attempt"] + 1, now + timeout_s
            db.execute("UPDATE items SET state = 'running', attempt = ?, deadline = ?, "
                       "started_at = ?, finished_at = NULL WHERE id = ?",
                       (attempt, deadline, now, row["id"]))
            db.execute("INSERT INTO attempts (item_id, attempt, started_at, deadline) "
                       "VALUES (?, ?, ?, ?)", (row["id"], attempt, now, deadline))
        return self.item(row["id"])

    def set_child_pid(self, item_id: int, attempt: int, pid: int) -> None:
        with self._tx() as db:
            db.execute("UPDATE attempts SET child_pid = ? WHERE item_id = ? AND attempt = ?",
                       (pid, item_id, attempt))

    def finish(self, item_id: int, attempt: int, result: Dict[str, Any],
               exit_code: Optional[int] = None) -> bool:
        """Persist ``result`` (a :func:`result_to_dict` dict). False if the item is not
        running this attempt (a stale write), and then nothing changes."""

        now = time.time()
        with self._tx() as db:
            cur = db.execute(
                "UPDATE items SET state = 'finished', status = ?, failure_kind = ?, "
                "result_json = ?, finished_at = ? WHERE id = ? AND attempt = ? AND state = 'running'",
                (result["status"], result["failure_kind"], json.dumps(result), now, item_id, attempt))
            if cur.rowcount != 1:
                return False
            db.execute("UPDATE attempts SET finished_at = ?, exit_code = ?, status = ?, "
                       "failure_kind = ?, message = ? WHERE item_id = ? AND attempt = ?",
                       (now, exit_code, result["status"], result["failure_kind"],
                        result["message"], item_id, attempt))
        return True

    def recover_running(self) -> List[int]:
        """Finish every item left ``running`` as an error / ``crash``; returns their ids.

        Call only when no worker is running (the worker calls it at start-up, holding
        its lock). Nothing is re-queued: that is W4's retry decision.
        """

        ids = []
        rows = self._db.execute("SELECT id, attempt FROM items WHERE state = 'running'").fetchall()
        for row in rows:
            result = result_to_dict(PathwayResult(
                status=STATUS_ERROR, stage=STAGE_INIT, failure_kind=KIND_CRASH,
                message=RECOVERED_MESSAGE.format(attempt=row["attempt"])))
            if self.finish(row["id"], row["attempt"], result):
                ids.append(row["id"])
        return ids

    # reads

    def item(self, item_id: int) -> Optional[Dict[str, Any]]:
        row = self._db.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["request"] = request_from_dict(json.loads(out.pop("request_json")))
        raw = out.pop("result_json")
        out["result"] = None if raw is None else json.loads(raw)
        out["attempts"] = [dict(a) for a in self._db.execute(
            "SELECT * FROM attempts WHERE item_id = ? ORDER BY attempt", (item_id,))]
        return out

    def submission(self, sid: str) -> Optional[Dict[str, Any]]:
        row = self._db.execute("SELECT * FROM submissions WHERE id = ?", (sid,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["consent"] = bool(out["consent"])
        out["items"] = [self.item(r["id"]) for r in self._db.execute(
            "SELECT id FROM items WHERE submission_id = ? ORDER BY position", (sid,))]
        return out
