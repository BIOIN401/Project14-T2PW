"""The job worker (W3): one process, one item at a time, each run in a child process.

Lifecycle of one item:

1. :meth:`JobStore.claim_next` moves the oldest queued item to ``running`` and sets
   its absolute ``deadline`` (now + ``timeout_s``) at dequeue.
2. The worker writes the request to ``<attempt dir>/request.json`` (``out_dir`` is
   always ``<attempt dir>/out``: the worker owns where an untrusted job writes) and
   starts ``python -m t2pw.web.worker --child <run_fn> <request.json> <result.json>``
   already contained:

   * Windows: created SUSPENDED, assigned to a new kill-on-close Job Object, and only
     then resumed, so no process of the run (a venv launcher's real interpreter
     included) can exist outside the job. If the job cannot be made, the child is
     killed before it ever runs and the item is ``error`` / ``crash``.
   * POSIX: ``start_new_session``, so the child leads a new process group from fork.

3. The child calls ``run_fn(request)`` (default ``t2pw.service.pathway:run_pathway``)
   and writes :func:`result_to_dict` to ``result.json``.
4. At the deadline the worker kills the child's tree: ``timeout`` / ``timeout``, unless
   a result was already written. A child that exits without a result is ``error`` /
   ``crash``. The tree is killed on every exit path, so a grandchild cannot outlive its
   run.

Only the tree this worker started is ever killed: the Job Object handle it created, or
the process group whose id is its own, still-unreaped child's PID. Never by image name.

Retry (W4): when an attempt ends, :func:`t2pw.web.retry.decide` says retry or final. A
retry puts the item back to ``queued``; the next claim opens attempt + 1 with a fresh
deadline, capped by the item's remaining wall budget.

Restart: :meth:`Worker.start` takes an exclusive lock on ``<db>.worker.lock`` (one
worker per database, whatever its work root), then :meth:`JobStore.recover_running`
finishes every item left ``running`` as ``error`` / ``crash``, and the same retry policy
decides whether that crash is re-queued. The orphaned child itself is already dead: on Windows the crashed worker's Job
Object handle closed with it; on POSIX the child watches its parent and kills its own
group when re-parented (and systemd's ``KillMode=control-group`` kills the cgroup).

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
from t2pw.web.retry import MAX_RETRIES, PER_ITEM_WALL_CAP_S, RUN_TIMEOUT_S, decide
DEFAULT_RUN_FN = "t2pw.service.pathway:run_pathway"
IS_WINDOWS = os.name == "nt"


class ContainmentError(RuntimeError):
    """The child could not be put in a killable tree; it was killed before running."""


# ── process-tree containment (pattern from evidence/bounded_run.py) ────────

if IS_WINDOWS:  # pragma: no cover - platform branch
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _ntdll = ctypes.WinDLL("ntdll")
    CREATE_SUSPENDED = 0x00000004

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

    def _new_job(process_handle: int) -> Optional[int]:
        """A kill-on-close Job Object holding the process, or None if that failed."""

        job = _k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = _ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = _k32.SetInformationJobObject(  # 9 = JobObjectExtendedLimitInformation
            wintypes.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info)) and \
            _k32.AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(process_handle))
        if not ok:
            _k32.CloseHandle(wintypes.HANDLE(job))
            return None
        return job


class _Child:
    """One contained child process tree."""

    def __init__(self, argv: List[str], log: IO[bytes]) -> None:
        kwargs: Dict[str, Any] = {"stdin": subprocess.DEVNULL, "stdout": log,
                                  "stderr": subprocess.STDOUT}
        if IS_WINDOWS:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | CREATE_SUSPENDED
        else:
            kwargs["start_new_session"] = True  # pgid == pid, from fork
        self.proc = subprocess.Popen(argv, **kwargs)  # noqa: S603 - argv is ours
        self.job: Optional[int] = None
        try:
            ok = True
            if IS_WINDOWS:
                # Popen's own process handle (full access, cannot name a recycled PID).
                handle = int(self.proc._handle)  # noqa: SLF001
                self.job = _new_job(handle)
                ok = self.job is not None and \
                    _ntdll.NtResumeProcess(wintypes.HANDLE(handle)) == 0
        except BaseException:  # e.g. KeyboardInterrupt: never leak a suspended child
            self.kill_tree()
            raise
        if not ok:
            self.kill_tree()  # still suspended: nothing of the run has executed
            raise ContainmentError("could not place the run in a Job Object")

    def wait_unreaped(self, deadline: float) -> bool:
        """Wait until the child exits or ``deadline``; True if it exited.

        POSIX leaves the exited child unreaped (``WNOWAIT``), so its PID, and with it
        our process-group id, cannot be reused before :meth:`kill_tree` signals it.
        """

        if IS_WINDOWS:
            try:
                self.proc.wait(timeout=max(0.0, deadline - time.time()))
                return True
            except subprocess.TimeoutExpired:
                return False
        flags = os.WEXITED | os.WNOHANG | os.WNOWAIT
        while time.time() < deadline:
            if os.waitid(os.P_PID, self.proc.pid, flags) is not None:
                return True
            time.sleep(0.2)
        return os.waitid(os.P_PID, self.proc.pid, flags) is not None

    def kill_tree(self) -> None:
        """Kill this child and every descendant still in its job / group, then reap."""

        if IS_WINDOWS:
            if self.job is not None:
                _k32.TerminateJobObject(wintypes.HANDLE(self.job), 1)
                _k32.CloseHandle(wintypes.HANDLE(self.job))
                self.job = None
            else:  # never resumed: only the suspended child itself exists
                self.proc.kill()
        else:
            try:  # the leader is not yet reaped, so this pgid is still ours
                os.killpg(self.proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        self.proc.wait()


# ── worker ─────────────────────────────────────────────────────────────────


def _crash(message: str) -> Dict[str, Any]:
    return result_to_dict(PathwayResult(status=STATUS_ERROR, stage=STAGE_INIT,
                                        failure_kind=KIND_CRASH, message=message))


class Worker:
    def __init__(self, store: JobStore, work_root: Path | str, *,
                 run_fn: str = DEFAULT_RUN_FN, timeout_s: float = RUN_TIMEOUT_S,
                 max_retries: int = MAX_RETRIES, wall_cap_s: float = PER_ITEM_WALL_CAP_S) -> None:
        self.max_retries = max_retries
        self.wall_cap_s = wall_cap_s
        self.store = store
        self.work_root = Path(work_root)
        self.run_fn = run_fn
        self.timeout_s = timeout_s
        self._lock: Optional[IO[bytes]] = None

    def start(self) -> List[int]:
        """Take the one-worker-per-database lock, then recover orphaned ``running`` items."""

        self.work_root.mkdir(parents=True, exist_ok=True)
        lock_path = Path(str(self.store.path) + ".worker.lock")
        fh = open(lock_path, "a+b")
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
            raise RuntimeError(f"another worker holds {lock_path}")
        self._lock = fh  # released by close(), or by the OS when this process dies
        return self.store.recover_running(self._should_retry)

    def _should_retry(self, item_id: int, attempt: int, result: Dict[str, Any]) -> bool:
        used = self.store.wall_used(item_id)
        return decide(result, attempt, used, max_retries=self.max_retries,
                      wall_cap_s=self.wall_cap_s).retry

    def _finish(self, item_id: int, attempt: int, result: Dict[str, Any],
                exit_code: Optional[int] = None) -> None:
        self.store.finish(item_id, attempt, result, exit_code,
                          requeue=self._should_retry(item_id, attempt, result))

    def close(self) -> None:
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    def run_one(self) -> Optional[int]:
        """Claim and run one item to completion. Returns its id, or None if none queued."""

        if self._lock is None:
            raise RuntimeError("call start() first")
        item = self.store.claim_next(self.timeout_s, self.wall_cap_s)
        if item is None:
            return None
        item_id, attempt, deadline = item["id"], item["attempt"], item["deadline"]
        item_dir = self.work_root / f"item-{item_id}" / f"attempt-{attempt}"
        item_dir.mkdir(parents=True, exist_ok=True)
        req = request_to_dict(item["request"])
        req.update(deadline=deadline, out_dir=str(item_dir / "out"))  # a submitted out_dir is ignored
        request_path, result_path = item_dir / "request.json", item_dir / "result.json"
        request_path.write_text(json.dumps(req), encoding="utf-8")
        argv = [sys.executable, "-m", "t2pw.web.worker", "--child", self.run_fn,
                str(request_path), str(result_path)]

        exited, child = False, None
        try:
            with open(item_dir / "child.log", "wb") as log:
                child = _Child(argv, log)
            self.store.set_child_pid(item_id, attempt, child.proc.pid)
            exited = child.wait_unreaped(deadline)
        except ContainmentError as exc:
            self._finish(item_id, attempt, _crash(str(exc)))
            return item_id
        finally:
            if child is not None:
                child.kill_tree()

        try:  # a result written before the deadline kill still counts
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            if exited:
                result = _crash(f"run process exited {child.proc.returncode} without a "
                                f"result; see {item_dir / 'child.log'}")
            else:
                result = result_to_dict(PathwayResult(
                    status=STATUS_TIMEOUT, stage=STAGE_INIT, failure_kind=KIND_TIMEOUT,
                    message=f"killed at the deadline ({deadline - item['started_at']:g} s after dequeue)"))
        self._finish(item_id, attempt, result, child.proc.returncode)
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


def _child_main(run_fn: str, request_path: str, result_path: str) -> int:
    _die_with_parent()
    try:
        req = request_from_dict(json.loads(Path(request_path).read_text(encoding="utf-8")))
        module, _, attr = run_fn.partition(":")
        res = getattr(importlib.import_module(module), attr)(req)
        if not isinstance(res, PathwayResult):
            raise TypeError(f"{run_fn} returned {type(res).__name__}, not PathwayResult")
        # default=str: an odd value in counts / attempt_log must not lose a finished run
        text = json.dumps(result_to_dict(res), default=str)
    except Exception as exc:  # noqa: BLE001 - every failure becomes a result
        traceback.print_exc()
        text = json.dumps(_crash(f"{type(exc).__name__}: {exc}"))
    tmp = result_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, result_path)
    return 0


def _exit_on_sigterm(signum: int, frame: Any) -> None:
    sys.exit(128 + signum)  # unwinds through run_one's finally: the child tree is killed


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--child"]:
        return _child_main(argv[1], argv[2], argv[3])
    ap = argparse.ArgumentParser(prog="python -m t2pw.web.worker")
    ap.add_argument("--db", required=True)
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--run-fn", default=DEFAULT_RUN_FN)
    ap.add_argument("--timeout", type=float, default=RUN_TIMEOUT_S)
    ap.add_argument("--poll", type=float, default=5.0)
    a = ap.parse_args(argv)
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    Worker(JobStore(a.db), a.work_root, run_fn=a.run_fn, timeout_s=a.timeout).run_forever(a.poll)
    return 0


if __name__ == "__main__":
    sys.exit(main())
