"""PathBank curator queue (W7, D-W02): a staging folder plus a table, never a PathBank write.

A consented, strict, ``release_ready`` result is copied into
``<queue_dir>/entry-<item_id>/`` (PWML, provenance reports, the citation report when one
exists, ``manifest.json``) and recorded in the ``curator_queue`` table of the jobs
database (this module creates it; W3's tables are untouched). A curator later lists the
pending entries and marks each ``imported`` or ``rejected`` with a note. Nothing here
talks to PathBank or any external database: importing is a human act.

Qualification (PO ruling Q-W0-3/5, ``docs/webserver/results/PO-RULINGS-2026-11-04.md``):
consent on, ``export_mode == strict``, ``release_status.status == release_ready``
(``pipeline/release_status.py:47``), and a PWML file that exists under the worker's
work root. Review-required and diagnostic-only results never qualify.

Idempotent: ``item_id`` is the table's primary key and the entry folder name is
deterministic, so a re-run, a retry or a restart enqueues once. A folder is built under
``.tmp-*`` with atomic per-file writes, then renamed into place; the row is inserted last.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from t2pw.pipeline.release_status import RELEASE_READY
from t2pw.service.pathway import MODE_STRICT, STATUS_PASS

log = logging.getLogger("t2pw.web.curator_queue")

PENDING, IMPORTED, REJECTED = "pending", "imported", "rejected"

#: ``report_paths`` keys copied as provenance: the run's own account of where each
#: element came from and which gates it passed. Why each (details in W7.md):
PROVENANCE_KEYS: Tuple[str, ...] = (
    "merged_payload.json",             # the evidence-bearing extraction the PWML is built from
    "final_mapped.json",               # entity identity / database ids of every element
    "rag_admission_report.json",       # which RAG-sourced facts were admitted, and why
    "contract_reports.json",           # stage-boundary contract results
    "final_stage3_gate_report.json",   # the biological gate verdict that released it
    "quarantine_report.json",          # what the strict run quarantined
    "removed_entity_report.json",      # what it removed (a curator must see omissions)
    "coverage_summary.json",           # completeness against the requested pathway
    "pwml_validation_report.json",     # the exported PWML validated
    "pwml_required_field_gate_report.json",  # required PathBank fields present
)
#: The citation report. Written only in research mode (W0 F5), so a strict result
#: normally has none; copied when present.
CITATION_KEYS: Tuple[str, ...] = ("research_pathway_citations.json",)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS curator_queue (
    item_id       INTEGER PRIMARY KEY,
    submission_id TEXT NOT NULL,
    organism      TEXT,
    pathway       TEXT,
    entry_dir     TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending', 'imported', 'rejected')),
    note          TEXT,
    created_at    REAL NOT NULL,
    decided_at    REAL
);
"""


def _under(path: Path, root: Optional[Path]) -> bool:
    return root is None or path.resolve().is_relative_to(root.resolve())


def qualify(submission: Dict[str, Any], item: Dict[str, Any],
            source_root: Optional[Path] = None) -> Tuple[bool, str]:
    """(qualifies, reason). ``item`` is a final ``JobStore.item`` dict."""

    if item.get("state") != "finished":
        return False, "not final"
    if not submission.get("consent"):
        return False, "no consent"
    if item["request"].export_mode != MODE_STRICT:
        return False, "not strict mode"
    res = item.get("result") or {}
    if res.get("status") != STATUS_PASS:
        return False, "status not pass"
    if (res.get("release_status") or {}).get("status") != RELEASE_READY:
        return False, "not release_ready"
    pwml = res.get("pwml_path")
    if not pwml or not Path(pwml).is_file() or not _under(Path(pwml), source_root):
        return False, "no PWML file"
    return True, "ok"


def _copy_atomic(src: Path, dst: Path) -> Dict[str, Any]:
    """Copy ``src`` to ``dst`` via ``dst.tmp`` + replace; returns size and sha256."""

    h, n = hashlib.sha256(), 0
    tmp = dst.with_name(dst.name + ".tmp")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        for chunk in iter(lambda: fi.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
            fo.write(chunk)
        fo.flush()
        os.fsync(fo.fileno())
    os.replace(tmp, dst)
    return {"bytes": n, "sha256": h.hexdigest()}


class CuratorQueue:
    def __init__(self, db_path: Path | str, queue_dir: Path | str) -> None:
        self.queue_dir = Path(queue_dir)
        self._db = sqlite3.connect(str(db_path), timeout=30, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    # ── write side ─────────────────────────────────────────────────────────

    def enqueue(self, submission: Dict[str, Any], item: Dict[str, Any],
                source_root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        """Stage a qualifying item. Returns its queue row, or None when it does not
        qualify. Calling it again for the same item returns the existing row."""

        ok, _ = qualify(submission, item, source_root)
        if not ok:
            return None
        existing = self.get(item["id"])
        if existing is not None:
            return existing
        scope = item["request"].scope
        name = f"entry-{item['id']}"
        final = self.queue_dir / name
        if final.exists() and not self._matches(final, submission, item):
            # ids restart after a DB reset: another job's folder. Move aside, never delete.
            os.rename(final, self.queue_dir / f".stale-{int(time.time())}-{uuid.uuid4().hex[:8]}-{name}")
        if not final.exists():
            self._stage(final, submission, item, source_root)
        self._db.execute(
            "INSERT OR IGNORE INTO curator_queue (item_id, submission_id, organism, pathway, "
            "entry_dir, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (item["id"], submission["id"], scope.requested_organism or None,
             scope.requested_pathway or None, name, time.time()))
        return self.get(item["id"])

    @staticmethod
    def _matches(entry: Path, submission: Dict[str, Any], item: Dict[str, Any]) -> bool:
        try:
            man = json.loads((entry / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return man.get("item_id") == item["id"] and man.get("submission_id") == submission["id"]

    def _stage(self, final: Path, submission: Dict[str, Any], item: Dict[str, Any],
               source_root: Optional[Path]) -> None:
        res, scope = item["result"], item["request"].scope
        tmp = self.queue_dir / f".tmp-{final.name}-{uuid.uuid4().hex}"
        try:
            files: List[Dict[str, Any]] = []
            missing: List[str] = []

            def add(role: str, src: Path, rel: str) -> None:
                files.append({"role": role, "path": rel, **_copy_atomic(src, tmp / rel)})

            add("pwml", Path(res["pwml_path"]), "pathway.pwml")
            reports = res.get("report_paths") or {}
            for role, keys, folder in (("provenance", PROVENANCE_KEYS, "provenance"),
                                       ("citation", CITATION_KEYS, "citations")):
                for key in keys:
                    src = Path(reports[key]) if reports.get(key) else None
                    if src is not None and src.is_file() and _under(src, source_root):
                        add(role, src, f"{folder}/{key}")
                    elif role == "provenance":
                        missing.append(key)
            manifest = {
                "schema": 1,
                "job_id": submission["id"],
                "submission_id": submission["id"],
                "item_id": item["id"],
                "organism": scope.requested_organism or None,
                "pathway_name": scope.requested_pathway or None,
                "export_mode": item["request"].export_mode,
                "release_status": res["release_status"],
                "consent": bool(submission["consent"]),
                "enqueued_at": time.time(),
                "files": files,
                "provenance_not_available": missing,
            }
            (tmp / "manifest.json.tmp").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            os.replace(tmp / "manifest.json.tmp", tmp / "manifest.json")
            os.rename(tmp, final)  # atomic publish of the whole entry
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)  # only our own .tmp-* folder
            raise

    # ── read / curator side ────────────────────────────────────────────────

    def get(self, item_id: int) -> Optional[Dict[str, Any]]:
        row = self._db.execute("SELECT * FROM curator_queue WHERE item_id = ?", (item_id,)).fetchone()
        return None if row is None else dict(row)

    def list_entries(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        sql, args = "SELECT * FROM curator_queue", ()
        if status is not None:
            sql, args = sql + " WHERE status = ?", (status,)
        return [dict(r) for r in self._db.execute(sql + " ORDER BY created_at, item_id", args)]

    def list_pending(self) -> List[Dict[str, Any]]:
        return self.list_entries(PENDING)

    def manifest(self, item_id: int) -> Optional[Dict[str, Any]]:
        row = self.get(item_id)
        if row is None:
            return None
        return json.loads((self.queue_dir / row["entry_dir"] / "manifest.json").read_text(encoding="utf-8"))

    def verify(self, item_id: int) -> bool:
        """True if every file in the entry still matches its manifest sha256."""

        row, man = self.get(item_id), self.manifest(item_id)
        if row is None or man is None:
            return False
        base = self.queue_dir / row["entry_dir"]
        try:
            return all(hashlib.sha256((base / f["path"]).read_bytes()).hexdigest() == f["sha256"]
                       for f in man["files"])
        except OSError:
            return False

    def _decide(self, item_id: int, status: str, note: str) -> bool:
        cur = self._db.execute(
            "UPDATE curator_queue SET status = ?, note = ?, decided_at = ? "
            "WHERE item_id = ? AND status = 'pending'", (status, note, time.time(), item_id))
        return cur.rowcount == 1

    def mark_imported(self, item_id: int, note: str = "") -> bool:
        """pending -> imported (the curator did the PathBank import by hand). False if not pending."""
        return self._decide(item_id, IMPORTED, note)

    def mark_rejected(self, item_id: int, note: str = "") -> bool:
        return self._decide(item_id, REJECTED, note)
