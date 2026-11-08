"""SQLite job store for the web server (W3).

A *submission* holds 1 to :data:`MAX_ITEMS_PER_SUBMISSION` *items*; each item is one
:class:`~t2pw.service.pathway.PathwayRequest`. The store also keeps the submission's
PathBank consent flag (default on) and optional email. Those two are stored only:
W7 and W8 act on them.

Item states: ``queued`` -> ``running`` -> ``finished``. ``finished`` says the item is
no longer being worked on; how it ended is the persisted result's ``status`` and
``failure_kind`` (the ``t2pw.service.pathway`` vocabulary). Every claim bumps the
item's ``attempt`` and opens one ``attempts`` row, so W4 can re-queue a finished item
(``state='queued'``) without a schema change. W4 does exactly that: ``finish(...,
requeue=True)`` (policy: ``t2pw.web.retry``).

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
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from t2pw.rag.eligibility import RequestedScope
from t2pw.service.pathway import (
    KIND_CRASH,
    STAGE_INIT,
    STATUS_ERROR,
    PathwayRequest,
    PathwayResult,
)

MAX_ITEMS_PER_SUBMISSION = 5

#: Total open items (queued + running) the store accepts. A typical run takes 20-60 min on one worker,
#: so 20 items is roughly 7-20 hours of backlog; the worst case is 20 x 90 min (the single try
#: cap, RUN_TIMEOUT_S) = 30 h. Open question Q-W3-1.
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

    def claim_next(self, timeout_s: float,
                   wall_cap_s: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Oldest queued item -> ``running``, next attempt, deadline = now + ``timeout_s``.

        The deadline is set here, at dequeue, so time spent queued never eats the
        run's wall-clock cap. With ``wall_cap_s`` it is also capped by what is left of
        the item's per-pathway wall budget (:meth:`wall_used`), so a retry can never
        run past it. A retried item keeps its id, so it is claimed before newer items.
        """

        with self._tx() as db:
            row = db.execute(
                "SELECT id, attempt FROM items WHERE state = 'queued' ORDER BY id LIMIT 1").fetchone()
            if row is None:
                return None
            now = time.time()
            if wall_cap_s is not None:
                timeout_s = min(timeout_s, max(0.0, wall_cap_s - self._wall_used(db, row["id"], now)))
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
               exit_code: Optional[int] = None, *, requeue: bool = False) -> bool:
        """Persist ``result`` (a :func:`result_to_dict` dict). False if the item is not
        running this attempt (a stale write), and then nothing changes.

        ``requeue=True`` (the retry policy said retry) closes the attempt's row as usual
        but puts the item back to ``queued`` with no item-level result: the failed
        attempt is history in ``attempts``, not the item's outcome.
        """

        now = time.time()
        with self._tx() as db:
            if requeue:
                cur = db.execute(
                    "UPDATE items SET state = 'queued', status = NULL, failure_kind = NULL, "
                    "result_json = NULL, finished_at = NULL "
                    "WHERE id = ? AND attempt = ? AND state = 'running'", (item_id, attempt))
            else:
                cur = db.execute(
                    "UPDATE items SET state = 'finished', status = ?, failure_kind = ?, "
                    "result_json = ?, finished_at = ? "
                    "WHERE id = ? AND attempt = ? AND state = 'running'",
                    (result["status"], result["failure_kind"], json.dumps(result), now,
                     item_id, attempt))
            if cur.rowcount != 1:
                return False
            db.execute("UPDATE attempts SET finished_at = ?, exit_code = ?, status = ?, "
                       "failure_kind = ?, message = ? WHERE item_id = ? AND attempt = ?",
                       (now, exit_code, result["status"], result["failure_kind"],
                        result["message"], item_id, attempt))
        return True

    def recover_running(self, should_retry: Optional[Callable[[int, int, Dict[str, Any]], bool]]
                        = None) -> List[int]:
        """Finish every item left ``running`` as an error / ``crash``; returns their ids.

        Call only when no worker is running (the worker calls it at start-up, holding
        its lock). Without ``should_retry`` nothing is re-queued. With it, called as
        ``should_retry(item_id, attempt, crash_result)``, a True answer re-queues the
        item (the crash stays in its attempts row); the worker passes the W4 policy.
        """

        ids = []
        rows = self._db.execute("SELECT id, attempt FROM items WHERE state = 'running'").fetchall()
        for row in rows:
            result = result_to_dict(PathwayResult(
                status=STATUS_ERROR, stage=STAGE_INIT, failure_kind=KIND_CRASH,
                message=RECOVERED_MESSAGE.format(attempt=row["attempt"])))
            again = bool(should_retry and should_retry(row["id"], row["attempt"], result))
            if self.finish(row["id"], row["attempt"], result, requeue=again):
                ids.append(row["id"])
        return ids

    # reads

    @staticmethod
    def _wall_used(db: sqlite3.Connection, item_id: int, now: float) -> float:
        # An attempt still open counts to ``now`` but never past its own deadline: the
        # worker kills at the deadline, so an attempt left open by a dead worker cannot
        # have run longer (it may have run less: downtime is then charged to the item).
        (used,) = db.execute(
            "SELECT COALESCE(SUM(MIN(COALESCE(finished_at, ?), COALESCE(deadline, ?)) - started_at), 0) "
            "FROM attempts WHERE item_id = ?", (now, now, item_id)).fetchone()
        return float(used)

    def wall_used(self, item_id: int, now: Optional[float] = None) -> float:
        """Seconds of run time this item has used across all its attempts (queue waits excluded)."""

        return self._wall_used(self._db, item_id, time.time() if now is None else now)

    def attempt_history(self, item_id: int) -> List[Dict[str, Any]]:
        """One dict per attempt, oldest first, for the job page (W6).

        Keys: ``attempt``, ``started_at``, ``finished_at`` (None while running),
        ``status``, ``failure_kind``, ``message``, ``retried`` (a later attempt
        exists), ``will_retry`` (it failed and the item is queued for the next one).
        """

        item = self._db.execute("SELECT state, attempt FROM items WHERE id = ?", (item_id,)).fetchone()
        if item is None:
            return []
        rows = self._db.execute(
            "SELECT attempt, started_at, finished_at, status, failure_kind, message "
            "FROM attempts WHERE item_id = ? ORDER BY attempt", (item_id,))
        return [dict(r, retried=r["attempt"] < item["attempt"],
                     will_retry=item["state"] == QUEUED and r["attempt"] == item["attempt"])
                for r in rows]

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
