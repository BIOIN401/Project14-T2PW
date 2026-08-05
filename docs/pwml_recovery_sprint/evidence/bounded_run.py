"""Bounded foreground process wrapper for the PWML recovery sprint.

Orchestration tooling, **not** pipeline code. Nothing under ``src/`` imports this
and this imports nothing from ``src/``. It exists to satisfy merge gate **G11**
(``TEST_MATRIX.md`` § 0, shared block ``[S8]``): every test, benchmark, pipeline
leg and LLM-backed command in this sprint runs through it, and a run that leaves
a surviving owned process is an *infrastructure failure*, not a test result.

Why it is not ``batch/runner.py``
---------------------------------
``runner.launch_child`` / ``_kill_tree`` (``runner.py:1107-1180``) already have the
**correct ownership model** -- they target only the PID they created and are not
global killers. Audited and confirmed. What they lack against ``[S8]``:

* cleanup runs only on the ``TimeoutExpired`` and ``KeyboardInterrupt`` paths.
  There is no ``finally``, so any other exception leaves the child alive.
* termination goes straight to ``taskkill /F`` -- no graceful stage.
* no post-kill survivor verification; no structured cleanup report.
* ``CREATE_NEW_PROCESS_GROUP`` is *not* a Job Object. It does not guarantee that
  descendants die with the root.
* it is used for batch legs only -- never for pytest runs, which are the largest
  memory consumers in this sprint.

``runner.py`` is owned by branch C-032 and **must not be modified here**. This is a
separate, orchestration-only wrapper that reuses the same ownership discipline.

Guarantees
----------
1. The child is placed in an isolated container: a Windows **Job Object** with
   ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``, or a POSIX process group.
2. An outer wall-clock timeout is enforced by the parent.
3. Cleanup runs in ``finally`` on **every** exit path -- success, nonzero,
   timeout, ``KeyboardInterrupt``, ``SystemExit``, or any other exception.
4. Termination is graceful first, forced after a grace period, and the report
   records whether the escalation was needed.
5. Survivors are **verified** by re-snapshotting the process table, not assumed.
6. The child's real exit code is returned unless cleanup verification itself
   failed.

Forbidden here, permanently: ``taskkill /IM``, ``pkill``, or any kill by image
name. Cleanup targets only PIDs this job created. Processes that already existed
when the job started are **reported**, never killed.

Usage
-----
::

    .venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/bounded_run.py \\
        --label smoke --timeout 900 -- \\
        .venv/Scripts/python.exe -m pytest -q --basetemp=<tmp>/smoke tests/...

The process exit code is the child's own. ``--json <path>`` also writes the
structured cleanup report required on every test record.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

IS_WINDOWS = os.name == "nt"

#: Exit code returned when the child finished but cleanup verification failed.
#: Distinct from any plausible pytest code so it can never be misread as a test
#: result. ``[S8]`` item 6: survivors are an infrastructure failure.
EXIT_INFRASTRUCTURE_FAILURE = 97

#: Seconds between "ask nicely" and "force". Deliberately short: this is cleanup
#: after the job is already over, not a shutdown hook worth waiting on.
DEFAULT_GRACE_SECONDS = 3.0

#: How long to keep re-checking the process table for survivors after forcing.
_SURVIVOR_SETTLE_SECONDS = 2.0
_POLL_INTERVAL = 0.1


# --------------------------------------------------------------------------- #
# Process-table access (no psutil in this venv -- ctypes on Windows, /proc on
# POSIX). Used for BOTH descendant discovery and survivor verification, so a
# survivor claim is measured rather than inferred.
# --------------------------------------------------------------------------- #

if IS_WINDOWS:  # pragma: no cover - platform branch
    import ctypes
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    TH32CS_SNAPPROCESS = 0x00000002
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
    JobObjectExtendedLimitInformation = 9
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    def snapshot_processes() -> Dict[int, Tuple[int, str]]:
        """``{pid: (ppid, image_name)}`` for every visible process."""

        snap = _kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        if snap == INVALID_HANDLE_VALUE or not snap:
            return {}
        out: Dict[int, Tuple[int, str]] = {}
        try:
            entry = PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
            if not _kernel32.Process32FirstW(snap, ctypes.byref(entry)):
                return {}
            while True:
                out[int(entry.th32ProcessID)] = (
                    int(entry.th32ParentProcessID),
                    str(entry.szExeFile),
                )
                if not _kernel32.Process32NextW(snap, ctypes.byref(entry)):
                    break
        finally:
            _kernel32.CloseHandle(snap)
        return out

else:  # POSIX

    def snapshot_processes() -> Dict[int, Tuple[int, str]]:
        """``{pid: (ppid, comm)}`` read from ``/proc``."""

        out: Dict[int, Tuple[int, str]] = {}
        proc_root = "/proc"
        if not os.path.isdir(proc_root):
            return out
        for name in os.listdir(proc_root):
            if not name.isdigit():
                continue
            try:
                with open(f"{proc_root}/{name}/stat", "r", encoding="utf-8") as fh:
                    raw = fh.read()
                # comm may contain spaces and parentheses; split on the LAST ')'.
                close = raw.rindex(")")
                comm = raw[raw.index("(") + 1 : close]
                rest = raw[close + 2 :].split()
                out[int(name)] = (int(rest[1]), comm)
            except (OSError, ValueError, IndexError):
                continue
        return out


def descendants_of(root_pid: int, table: Dict[int, Tuple[int, str]]) -> List[int]:
    """Every transitive child of *root_pid* in *table*, excluding the root."""

    children: Dict[int, List[int]] = {}
    for pid, (ppid, _name) in table.items():
        children.setdefault(ppid, []).append(pid)
    found: List[int] = []
    stack = list(children.get(root_pid, ()))
    seen = {root_pid}
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        found.append(pid)
        stack.extend(children.get(pid, ()))
    return sorted(found)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class CleanupReport:
    """The record ``[S8]`` item 9 requires on every test entry."""

    label: str
    command: List[str]
    cwd: str
    started_at: str
    finished_at: str = ""
    duration_seconds: float = 0.0
    timeout_seconds: float = 0.0
    root_pid: Optional[int] = None
    isolation: str = "none"
    process_group: Optional[int] = None
    exit_reason: str = "unknown"
    exit_code: Optional[int] = None
    returned_code: Optional[int] = None
    descendants_observed: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    descendants_terminated: List[int] = dataclasses.field(default_factory=list)
    graceful_attempted: bool = False
    forced: bool = False
    final_surviving_count: int = 0
    survivors: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    preexisting_reported: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    cleanup_success: bool = False
    notes: List[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def render(self) -> str:
        lines = [
            "",
            "================ CLEANUP REPORT ================",
            f"label                   : {self.label}",
            f"command                 : {' '.join(self.command)}",
            f"cwd                     : {self.cwd}",
            f"root PID / process group: {self.root_pid} / {self.process_group}",
            f"isolation               : {self.isolation}",
            f"timeout                 : {self.timeout_seconds:g} s",
            f"duration                : {self.duration_seconds:.2f} s",
            f"exit reason             : {self.exit_reason}",
            f"exit code (real)        : {self.exit_code}",
            f"returned code           : {self.returned_code}",
            f"descendants observed    : {len(self.descendants_observed)}"
            + (
                "  " + ", ".join(f"{d['pid']}:{d['name']}" for d in self.descendants_observed)
                if self.descendants_observed
                else ""
            ),
            f"descendants terminated  : {len(self.descendants_terminated)}"
            + ("  " + ", ".join(str(p) for p in self.descendants_terminated)
               if self.descendants_terminated else ""),
            f"graceful attempted      : {self.graceful_attempted}",
            f"forced                  : {self.forced}",
            f"FINAL SURVIVING COUNT   : {self.final_surviving_count}",
            f"cleanup                 : {'success' if self.cleanup_success else 'FAILURE'}",
        ]
        for surv in self.survivors:
            lines.append(f"  SURVIVOR pid={surv.get('pid')} name={surv.get('name')} "
                         f"ppid={surv.get('ppid')} rss={surv.get('rss_mb')}")
        if self.preexisting_reported:
            lines.append(
                f"pre-existing (reported, NEVER killed): {len(self.preexisting_reported)}"
            )
        for note in self.notes:
            lines.append(f"note                    : {note}")
        lines.append("================================================")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Windows Job Object
# --------------------------------------------------------------------------- #


class _JobObject:
    """A Windows Job Object set to kill its members when the handle closes.

    Assignment happens immediately after ``Popen`` returns. The interpreter has
    not finished initialising by then, so it cannot have spawned anything that
    escapes. If assignment fails the caller falls back to the process-group +
    owned-PID model and the report records that.
    """

    def __init__(self, name_hint: str) -> None:  # pragma: no cover - platform branch
        self.handle = None
        self.ok = False
        if not IS_WINDOWS:
            return
        handle = _kernel32.CreateJobObjectW(None, None)
        if not handle:
            return
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = _kernel32.SetInformationJobObject(
            ctypes.c_void_p(handle),
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            _kernel32.CloseHandle(ctypes.c_void_p(handle))
            return
        self.handle = handle
        self.ok = True

    def assign(self, pid: int) -> bool:  # pragma: no cover - platform branch
        if not self.ok:
            return False
        PROCESS_SET_QUOTA, PROCESS_TERMINATE = 0x0100, 0x0001
        hproc = _kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
        if not hproc:
            return False
        try:
            return bool(
                _kernel32.AssignProcessToJobObject(
                    ctypes.c_void_p(self.handle), ctypes.c_void_p(hproc)
                )
            )
        finally:
            _kernel32.CloseHandle(ctypes.c_void_p(hproc))

    def terminate(self) -> None:  # pragma: no cover - platform branch
        if self.ok:
            _kernel32.TerminateJobObject(ctypes.c_void_p(self.handle), 1)

    def close(self) -> None:  # pragma: no cover - platform branch
        if self.ok and self.handle:
            _kernel32.CloseHandle(ctypes.c_void_p(self.handle))
            self.handle, self.ok = None, False


# --------------------------------------------------------------------------- #
# Termination -- OWNED PIDS ONLY
# --------------------------------------------------------------------------- #


def _graceful(root_pid: int, owned: Sequence[int], report: CleanupReport) -> None:
    """Ask nicely. Windows: ``taskkill /T`` WITHOUT ``/F``. POSIX: ``SIGTERM``."""

    report.graceful_attempted = True
    if IS_WINDOWS:
        # /T walks the tree; no /F means WM_CLOSE / CTRL, not an immediate kill.
        # The PID is one we created -- never an image name.
        try:
            subprocess.run(
                ["taskkill", "/PID", str(root_pid), "/T"],
                capture_output=True, timeout=30, check=False,
            )
        except Exception as exc:  # noqa: BLE001
            report.notes.append(f"graceful taskkill raised {exc!r}")
        return
    try:
        os.killpg(os.getpgid(root_pid), signal.SIGTERM)
    except Exception:  # noqa: BLE001 - already gone
        for pid in owned:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:  # noqa: BLE001
                pass


def _force(job: Optional[_JobObject], root_pid: int, owned: Sequence[int],
           report: CleanupReport) -> None:
    """Forced termination of the job / owned PIDs. Still never by image name."""

    report.forced = True
    if job is not None and job.ok:
        job.terminate()
    if IS_WINDOWS:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(root_pid)],
                capture_output=True, timeout=30, check=False,
            )
        except Exception as exc:  # noqa: BLE001
            report.notes.append(f"forced taskkill raised {exc!r}")
        for pid in owned:
            if pid == root_pid:
                continue
            try:
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    capture_output=True, timeout=15, check=False,
                )
            except Exception:  # noqa: BLE001
                pass
        return
    try:
        os.killpg(os.getpgid(root_pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001
        pass
    for pid in owned:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass


def _still_alive(owned: Dict[int, str]) -> List[Dict[str, Any]]:
    """Re-snapshot and report which owned PIDs are genuinely still present.

    Matching on ``(pid, image name)`` rather than PID alone, so a recycled PID
    belonging to some unrelated process is not misreported as a survivor.
    """

    table = snapshot_processes()
    out: List[Dict[str, Any]] = []
    for pid, name in owned.items():
        entry = table.get(pid)
        if entry is None:
            continue
        ppid, live_name = entry
        if name and live_name and live_name.lower() != name.lower():
            continue  # PID reused by an unrelated process
        out.append({"pid": pid, "name": live_name, "ppid": ppid, "rss_mb": None})
    return out


# --------------------------------------------------------------------------- #
# The wrapper
# --------------------------------------------------------------------------- #


def run(
    command: Sequence[str],
    *,
    timeout: float,
    label: str = "job",
    cwd: Optional[str] = None,
    grace: float = DEFAULT_GRACE_SECONDS,
    echo: bool = True,
    env: Optional[Dict[str, str]] = None,
) -> CleanupReport:
    """Run *command* in the foreground, bounded, with guaranteed cleanup."""

    command = [str(part) for part in command]
    cwd = cwd or os.getcwd()
    report = CleanupReport(
        label=label,
        command=command,
        cwd=cwd,
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        timeout_seconds=float(timeout),
    )

    # Pre-existing processes are REPORTED, never killed ([S8] item 4). Snapshot
    # before the child starts so ownership is unambiguous afterwards.
    before = snapshot_processes()
    started = time.monotonic()

    job: Optional[_JobObject] = None
    proc: Optional[subprocess.Popen] = None
    owned: Dict[int, str] = {}
    log_handle = None
    log_path = None
    read_cursor = 0

    # A file, not a pipe: a pipe can deadlock on a full buffer, and the whole
    # point of this wrapper is that the parent never blocks unboundedly.
    fd, log_path = tempfile.mkstemp(prefix=f"boundedrun_{label}_", suffix=".log")
    os.close(fd)

    def _drain() -> None:
        nonlocal read_cursor
        if not echo:
            return
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
                fh.seek(read_cursor)
                chunk = fh.read()
                read_cursor = fh.tell()
            if chunk:
                sys.stdout.write(chunk)
                sys.stdout.flush()
        except OSError:
            pass

    try:
        kwargs: Dict[str, Any] = {"cwd": cwd, "env": env or os.environ.copy()}
        if IS_WINDOWS:
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["start_new_session"] = True

        log_handle = open(log_path, "wb")
        proc = subprocess.Popen(  # noqa: S603 - argv is ours
            command, stdout=log_handle, stderr=subprocess.STDOUT, **kwargs
        )
        report.root_pid = proc.pid
        owned[proc.pid] = ""

        if IS_WINDOWS:
            job = _JobObject(label)
            if job.assign(proc.pid):
                report.isolation = "windows_job_object(KILL_ON_JOB_CLOSE)"
            else:
                report.isolation = "windows_process_group(fallback)"
                report.notes.append(
                    "Job Object assignment failed; fell back to owned-PID tree kill"
                )
        else:
            try:
                report.process_group = os.getpgid(proc.pid)
            except Exception:  # noqa: BLE001
                report.process_group = None
            report.isolation = "posix_process_group"

        # Record the root's real image name now that it exists, so survivor
        # verification can distinguish it from a recycled PID.
        table = snapshot_processes()
        if proc.pid in table:
            owned[proc.pid] = table[proc.pid][1]

        # ---- bounded wait; interruptible so `finally` always runs ----
        deadline = started + float(timeout)
        timed_out = False
        while True:
            rc = proc.poll()
            _drain()
            if rc is not None:
                break
            live = snapshot_processes()
            for pid in descendants_of(proc.pid, live):
                if pid not in owned and pid not in before:
                    owned[pid] = live[pid][1]
                    report.descendants_observed.append(
                        {"pid": pid, "name": live[pid][1], "ppid": live[pid][0]}
                    )
            if time.monotonic() >= deadline:
                timed_out = True
                break
            time.sleep(_POLL_INTERVAL)

        if timed_out:
            report.exit_reason = "timeout"
            report.exit_code = None
        else:
            report.exit_code = proc.returncode
            report.exit_reason = "completed" if proc.returncode == 0 else "nonzero"

    except KeyboardInterrupt:
        report.exit_reason = "cancelled"
        report.notes.append("KeyboardInterrupt in the parent; cleanup ran via finally")
    except BaseException as exc:  # noqa: BLE001 - cleanup must run for ANY exit
        report.exit_reason = "cancelled"
        report.notes.append(f"parent raised {type(exc).__name__}: {exc}")
        raise
    finally:
        # ------------------------------------------------------------------ #
        # CLEANUP -- every exit path reaches here, including cancellation.
        # ------------------------------------------------------------------ #
        _drain()
        if log_handle is not None:
            try:
                log_handle.close()
            except OSError:
                pass

        if proc is not None:
            # Capture descendants one last time before killing anything.
            live = snapshot_processes()
            for pid in descendants_of(proc.pid, live):
                if pid not in owned and pid not in before:
                    owned[pid] = live[pid][1]
                    report.descendants_observed.append(
                        {"pid": pid, "name": live[pid][1], "ppid": live[pid][0]}
                    )

            outstanding = _still_alive(owned)
            if outstanding:
                _graceful(proc.pid, list(owned), report)
                grace_deadline = time.monotonic() + max(0.0, float(grace))
                while time.monotonic() < grace_deadline:
                    if not _still_alive(owned):
                        break
                    time.sleep(_POLL_INTERVAL)
                if _still_alive(owned):
                    _force(job, proc.pid, list(owned), report)

            if job is not None:
                job.close()  # KILL_ON_JOB_CLOSE: last-resort sweep of the job

            try:
                proc.poll()
            except Exception:  # noqa: BLE001
                pass

            # VERIFY, do not assume. Re-check until the table settles.
            settle_deadline = time.monotonic() + _SURVIVOR_SETTLE_SECONDS
            survivors = _still_alive(owned)
            while survivors and time.monotonic() < settle_deadline:
                time.sleep(_POLL_INTERVAL)
                survivors = _still_alive(owned)

            terminated = [
                pid for pid in owned if pid not in {s["pid"] for s in survivors}
            ]
            report.descendants_terminated = sorted(
                pid for pid in terminated if pid != proc.pid
            )
            report.survivors = survivors
            report.final_surviving_count = len(survivors)
            report.cleanup_success = not survivors

        report.duration_seconds = time.monotonic() - started
        report.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        # Pre-existing processes: reported only. NEVER killed.
        report.preexisting_reported = [
            {"pid": pid, "name": before[pid][1]}
            for pid in sorted(before)
            if before[pid][1].lower() in {"python.exe", "streamlit.exe", "pytest.exe"}
        ]

        if not report.cleanup_success:
            report.exit_reason = "infrastructure_failure"
            report.returned_code = EXIT_INFRASTRUCTURE_FAILURE
        elif report.exit_reason == "timeout":
            report.returned_code = 124
        elif report.exit_reason == "cancelled":
            report.returned_code = 130
        else:
            report.returned_code = report.exit_code

        if log_path and os.path.exists(log_path):
            try:
                os.unlink(log_path)
            except OSError:
                pass

    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bounded foreground process wrapper (sprint gate G11 / [S8])."
    )
    parser.add_argument("--timeout", type=float, required=True, help="outer wall clock, seconds")
    parser.add_argument("--label", default="job")
    parser.add_argument("--cwd", default=None)
    parser.add_argument("--grace", type=float, default=DEFAULT_GRACE_SECONDS)
    parser.add_argument("--json", dest="json_path", default=None,
                        help="write the structured cleanup report here")
    parser.add_argument("--quiet", action="store_true", help="do not echo child output")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(list(argv) if argv is not None else None)

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("no command given (use: ... --timeout N -- <command>)")

    report = run(
        command,
        timeout=args.timeout,
        label=args.label,
        cwd=args.cwd,
        grace=args.grace,
        echo=not args.quiet,
    )
    print(report.render(), file=sys.stderr)
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2)
    return int(report.returned_code if report.returned_code is not None else 1)


if __name__ == "__main__":
    raise SystemExit(main())
