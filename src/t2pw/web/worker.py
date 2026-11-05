"""The job worker (W3): one process, one item at a time, each run in a child process.

Lifecycle of one item:

1. :meth:`JobStore.claim_next` moves the oldest queued item to ``running`` and sets
   its absolute ``deadline`` (now + ``timeout_s``) at dequeue.
2. The worker starts ``python -m t2pw.web.worker --child <run_fn> <result.json>`` in
   its own process tree: a Windows Job Object (kill-on-close), or a POSIX session /
   process group. The request goes to the child on stdin *after* the child is
   contained, so the run cannot start before the tree is owned.
3. The child calls ``run_fn(request)`` (default ``t2pw.service.pathway:run_pathway``)
   and writes :func:`result_to_dict` to ``result.json``.
4. At the deadline the worker kills the child's tree: ``timeout`` / ``timeout``. A
   child that exits without a result is an ``error`` / ``crash``. The tree is killed
   on every exit path, so a grandchild cannot outlive its run.

Only the tree this worker started is ever killed: a Job Object handle it created, a
process group whose id is its own child's PID, or ``taskkill /T /PID <its child>``.
Never by image name.

Restart: :meth:`Worker.start` takes an exclusive lock file, then
:meth:`JobStore.recover_running` finishes every item left ``running`` as ``error`` /
``crash``. No retry here: that is W4. The orphaned child itself is already dead: on
Windows the crashed worker's Job Object handle closed with it; on POSIX the child
watches its parent and kills its own group when re-parented (and systemd's
``KillMode=control-group`` kills the unit's cgroup anyway).

Must not import ``streamlit`` or ``t2pw.app``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Sequence

from t2pw.service.pathway import (
    KIND_CRASH,
    KIND_TIMEOUT,
    STAGE_INIT,
    STATUS_ERROR,
    STATUS_TIMEOUT,
    PathwayResult,
)
from t2pw.web.jobs import JobStore, request_from_dict, request_to_dict, result_to_dict

#: Wall-clock cap per item, from dequeue. The batch driver's leg timeout is also 60 min.
#: W4 owns the final number (D-W07).
RUN_TIMEOUT_S = 60 * 60
DEFAULT_RUN_FN = "t2pw.service.pathway:run_pathway"
IS_WINDOWS = os.name == "nt"


# ── process-tree containment (pattern from evidence/bounded_run.py) ────────

if IS_WINDOWS:  # pragma: no cover - platform branch
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.OpenProcess.restype = wintypes.HANDLE

    class _BasicLimits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                    ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BasicLimits),
                    ("IoInfo", ctypes.c_ulonglong * 6),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    def _new_job(pid: int) -> Optional[int]:
        """A kill-on-close Job Object holding ``pid``, or None if that failed."""

        job = _k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = _ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = _k32.SetInformationJobObject(  # 9 = JobObjectExtendedLimitInformation
            wintypes.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info))
        proc = _k32.OpenProcess(0x0100 | 0x0001, False, pid)  # SET_QUOTA | TERMINATE
        if proc:
            ok = ok and _k32.AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(proc))
            _k32.CloseHandle(wintypes.HANDLE(proc))
        ok = ok and bool(proc)
        if not ok:
            _k32.CloseHandle(wintypes.HANDLE(job))
            return None
        return job


class _Child:
    """One contained child process tree."""

    def __init__(self, argv: List[str], log: IO[bytes]) -> None:
        kwargs: Dict[str, Any] = {"stdin": subprocess.PIPE, "stdout": log,
                                  "stderr": subprocess.STDOUT}
        if IS_WINDOWS:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True  # pgid == pid
        self.proc = subprocess.Popen(argv, **kwargs)  # noqa: S603 - argv is ours
        self.job = _new_job(self.proc.pid) if IS_WINDOWS else None

    def kill_tree(self) -> None:
        """Kill this child and every descendant still in its job / group."""

        if IS_WINDOWS:
            if self.job is not None:
                _k32.TerminateJobObject(wintypes.HANDLE(self.job), 1)
                _k32.CloseHandle(wintypes.HANDLE(self.job))
                self.job = None
            elif self.proc.poll() is None:  # fallback: walk the tree by OUR child's PID
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                               capture_output=True, timeout=30, check=False)
        else:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass  # group already empty
        self.proc.wait()


# ── worker ─────────────────────────────────────────────────────────────────


class Worker:
    def __init__(self, store: JobStore, work_root: Path | str, *,
                 run_fn: str = DEFAULT_RUN_FN, timeout_s: float = RUN_TIMEOUT_S) -> None:
        self.store = store
        self.work_root = Path(work_root)
        self.run_fn = run_fn
        self.timeout_s = timeout_s
        self._lock: Optional[IO[bytes]] = None

    def start(self) -> List[int]:
        """Take the single-worker lock, then recover orphaned ``running`` items."""

        self.work_root.mkdir(parents=True, exist_ok=True)
        fh = open(self.work_root / "worker.lock", "a+b")
        try:
            if IS_WINDOWS:
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            raise RuntimeError(f"another worker holds {self.work_root / 'worker.lock'}")
        self._lock = fh  # released by close(), or by the OS when this process dies
        return self.store.recover_running()

    def close(self) -> None:
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    def run_one(self) -> Optional[int]:
        """Claim and run one item to completion. Returns its id, or None if none queued."""

        if self._lock is None:
            raise RuntimeError("call start() first")
        item = self.store.claim_next(self.timeout_s)
        if item is None:
            return None
        item_id, attempt, deadline = item["id"], item["attempt"], item["deadline"]
        item_dir = self.work_root / f"item-{item_id}" / f"attempt-{attempt}"
        item_dir.mkdir(parents=True, exist_ok=True)
        req = item["request"]
        req = replace(req, deadline=deadline, out_dir=req.out_dir or item_dir / "out")
        result_path = item_dir / "result.json"
        argv = [sys.executable, "-m", "t2pw.web.worker", "--child", self.run_fn, str(result_path)]

        timed_out, child = False, None
        try:
            with open(item_dir / "child.log", "wb") as log:
                child = _Child(argv, log)
            self.store.set_child_pid(item_id, attempt, child.proc.pid)
            try:
                child.proc.stdin.write(json.dumps(request_to_dict(req)).encode("utf-8"))
                child.proc.stdin.close()
            except OSError:
                pass  # child died before reading; reported as a crash below
            try:
                child.proc.wait(timeout=max(0.0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                timed_out = True
        finally:
            if child is not None:
                child.kill_tree()

        exit_code = child.proc.returncode
        if timed_out:
            result = result_to_dict(PathwayResult(
                status=STATUS_TIMEOUT, stage=STAGE_INIT, failure_kind=KIND_TIMEOUT,
                message=f"killed at the deadline ({self.timeout_s:g} s after dequeue)"))
        else:
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                result = result_to_dict(PathwayResult(
                    status=STATUS_ERROR, stage=STAGE_INIT, failure_kind=KIND_CRASH,
                    message=f"run process exited {exit_code} without a result; see {item_dir / 'child.log'}"))
        self.store.finish(item_id, attempt, result, exit_code=exit_code)
        return item_id

    def run_forever(self, poll_s: float = 5.0) -> None:
        self.start()
        try:
            while True:
                if self.run_one() is None:
                    time.sleep(poll_s)
        finally:
            self.close()


# ── child side ─────────────────────────────────────────────────────────────


def _die_with_parent() -> None:
    """POSIX: kill our own process group if the worker dies (we get re-parented).

    Windows needs nothing: the worker's Job Object handle closes when it dies.
    """

    if IS_WINDOWS:
        return
    parent = os.getppid()

    def watch() -> None:
        while os.getppid() == parent:
            time.sleep(0.5)
        os.killpg(os.getpgrp(), signal.SIGKILL)

    threading.Thread(target=watch, daemon=True).start()


def _child_main(run_fn: str, result_path: str) -> int:
    _die_with_parent()
    req = request_from_dict(json.loads(sys.stdin.buffer.read()))
    try:
        module, _, attr = run_fn.partition(":")
        res = getattr(importlib.import_module(module), attr)(req)
        if not isinstance(res, PathwayResult):
            raise TypeError(f"{run_fn} returned {type(res).__name__}, not PathwayResult")
        out = result_to_dict(res)
    except Exception as exc:  # noqa: BLE001 - every failure becomes a result
        traceback.print_exc()
        out = result_to_dict(PathwayResult(status=STATUS_ERROR, stage=STAGE_INIT,
                                           failure_kind=KIND_CRASH,
                                           message=f"{type(exc).__name__}: {exc}"))
    tmp = result_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    os.replace(tmp, result_path)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--child"]:
        return _child_main(argv[1], argv[2])
    ap = argparse.ArgumentParser(prog="python -m t2pw.web.worker")
    ap.add_argument("--db", required=True)
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--run-fn", default=DEFAULT_RUN_FN)
    ap.add_argument("--timeout", type=float, default=RUN_TIMEOUT_S)
    ap.add_argument("--poll", type=float, default=5.0)
    a = ap.parse_args(argv)
    Worker(JobStore(a.db), a.work_root, run_fn=a.run_fn, timeout_s=a.timeout).run_forever(a.poll)
    return 0


if __name__ == "__main__":
    sys.exit(main())
